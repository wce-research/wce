"""Démo : Invariant → transformations stabilisatrices ; Transformation → invariants préservés
(dualité « Galois-inspired » établie EMPIRIQUEMENT — pas une correspondance de Galois stricte)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.invariant_algebra import (CODE_SAMPLES, EquivalenceClass, LANG_INVARIANTS, TransformationOrbit, algebra_key,  # noqa: E402
                                    algebra_transformations, build_index, canonical_equation, canonical_program,
                                    canonical_sentence, linear_equation, run_program)
from core.perturbation_benchmark import compression_experiment  # noqa: E402

if __name__ == "__main__":
    ix = build_index("algebra")
    print("ALGÈBRE — Invariant solution_set -> Stabilizer :", sorted(ix.stabilizer("solution_set").transformations))
    for t in ("SquareBothSides", "MultiplyBothSides(0)", "MultiplyBothSides(x)", "SwapSides"):
        print(f"  Transformation {t:22s} -> préserve {sorted(ix.preserves(t))}")
    e = linear_equation(2, 3, 7)
    G = [t for t in algebra_transformations() if t.name in ix.stabilizer("solution_set")][:6]
    orb = TransformationOrbit(e, G, algebra_key, max_states=300, max_depth=2)
    print(f"  Orbite de « {e} » sous 6 transformations admissibles (profondeur 2) : {len(orb)} états ; forme canonique : {canonical_equation(e)}")
    lx = build_index("language")
    print("\nLANGAGE —", {t: sorted(v) for t, v in lx.matrix().items()})
    ec = EquivalenceClass(LANG_INVARIANTS, ["semantic_action"])
    for s in ("la balle tombe", "la balle chute", "le bloc tombe", "le cube roule"): ec.add(s)
    print("  classes selon l'action sémantique :", {str(k.as_dict()): v for k, v in ec.classes.items()})
    ec2 = EquivalenceClass(LANG_INVARIANTS, ["semantic_meaning"])
    for s in ("la balle tombe", "la balle chute", "le bloc tombe"): ec2.add(s)
    print("  classes selon le sens (entité + action) :", [v for v in ec2.classes.values()], "; canonique de « la boule dégringole » :", canonical_sentence("la boule dégringole"))
    cx = build_index("code")
    print("\nCODE —", {t: sorted(v) for t, v in cx.matrix().items()})
    p = CODE_SAMPLES[0]; print(f"  {p} -> sortie {run_program(p)} ; normalisé : {canonical_program(p)}")
    for d in ("sequence", "geometry"):
        print(f"\n{d.upper()} —", {t: sorted(v) for t, v in build_index(d).matrix().items()})
    print("\nCOMPRESSION (formes canoniques + chemins vs tous les exemples) :")
    for d, r in compression_experiment(0).items(): print(f"  {d:9s} {r}")

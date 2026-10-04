"""Démo : nouvelle tâche → adaptation candidate (poids rapides) → contrôles d'invariants → consolider ou rollback."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.safe_plasticity import METHODS, make_invariants, old_model, run_method, safe_consolidate, task_data, train_fast  # noqa: E402
import copy  # noqa: E402

if __name__ == "__main__":
    base = old_model(0); inv = make_invariants(base, 0)
    print("Modèle ancien : exactitudes", {k: round(v, 3) for k, v in inv.reference.items()})
    xb, yb, xbv, ybv = task_data("B", 100)
    m = copy.deepcopy(base); mods = tuple(m.topological_order())
    print("\n1) Adaptation candidate SANS contrainte (poids rapides), puis contrôle :")
    rep = safe_consolidate(m, train_fast(m, mods, xb, yb, 600), xbv, ybv, inv, base.accuracy(xbv, ybv))
    print(f"   décision : {rep.decision} ; gain {rep.task_gain:+.3f} ; raisons : {rep.reasons}")
    print(f"   rétention après décision : {inv.retention(m)} (poids lents intacts si rollback)")
    print("\n2) Adaptation candidate CONTRAINTE par l'invariant d'équivalence de sortie (ancres non étiquetées) :")
    m = copy.deepcopy(base)
    import torch
    ox = torch.cat([task_data(t, 0)[0][:300] for t in "AC"])
    with torch.no_grad(): an = (ox, m.forward(ox))
    rep = safe_consolidate(m, train_fast(m, mods, xb, yb, 600, anchors=an, lam=0.3), xbv, ybv, inv, base.accuracy(xbv, ybv))
    print(f"   décision : {rep.decision} ; gain {rep.task_gain:+.3f} ; rayon avant/après {rep.radius_before:.3f}/{rep.radius_after:.3f} ; interaction {rep.dangerous_interaction:+.3f} ; {rep.reasons}")
    print("\n3) Comparaison des 5 méthodes (graine 0) :")
    for meth in METHODS:
        r = run_method(meth, 0, base=base)
        print(f"   {meth:22s} nouvelle {r['new_task']:.3f} | rétention {r['retention']:.3f} | oubli {r['forgetting']:.3f} | params changés {r['parameters_changed']:5d} | rollbacks {r['rollbacks']} | {r['consolidation']}")

"""Démo : nouvelle tâche → tentative de réutilisation → transformations admissibles → adaptation locale ou
croissance → vérification des invariants (certificats) → consolidation. Aucun identifiant de tâche."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.compositional_plasticity import PlasticityPlanner, StrategyLibrary, induce_macros  # noqa: E402
from core.skillnet import SkillNet, task_stream  # noqa: E402

if __name__ == "__main__":
    tasks = task_stream(1, 12); rng = random.Random(0); net = SkillNet(); certs = {}; lib = StrategyLibrary()
    pl = PlasticityPlanner(library=lib)
    print("Séquence (la spécification est CACHÉE au système ; seuls des exemples bruités sont vus) :")
    for i, t in enumerate(tasks):
        X, C, y = t.sample(128, rng); rec = pl.learn(net, X, C, y, certs)
        ok = all(c.check(net)[0] for c in certs.values())
        print(f"  {t.name:4s} [{t.kind:10s}] contexte {rec['context']:6s} {'nouveau' if rec['new_context'] else 'connu  '} -> "
              f"{rec['kind']:15s} {str(rec.get('program', '')):60s} params {net.parameter_count():5d} | certificats {len(certs)} OK={ok} | rollbacks {rec['rollback']}")
    tests = [t.test_set(100 + i) for i, t in enumerate(tasks)]
    print(f"\nExactitude finale moyenne sur les {len(tasks)} tâches : {sum(net.accuracy(*ts) for ts in tests) / len(tests):.3f}")
    print("Compétences :", {s: p if p[0] != 'adapter' else ('adapter', p[1]) for s, p in net.skills.items()})
    print("Macros induites par anti-unification :")
    for m in induce_macros(pl.programs): print(f"   {m.name:45s} support {m.support} variables {m.variables}")
    print("Historique TransformGraph (10 dernières) :", net.history()[-10:])

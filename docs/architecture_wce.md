# WCE — World Configuration Engine

WCE est un paradigme expérimental alternatif : des primitives, transformations, invariants et compositions pour explorer la construction de structures et de comportements complexes. Ce document décrit les mécanismes scientifiques ; les implémentations du site et du futur hub ne font pas partie de cette publication.

## What exists now — le socle vérifiable

Le moteur symbolique suit un parcours explicite :

```text
language/command.py → core/executor.py → core/world.py
                                      → physics/engine.py
                                      → language/answer.py
```

`app/session.py` assemble un monde réduit avec une balle et un support. `app/repl.py` permet de manipuler ce moteur. Les lois de gravité et de collision sont définies à l’avance : la démonstration ne les découvre pas.

Le dépôt comprend également des prototypes scientifiques, des tests et des protocoles. Trois artefacts consultables décrivent la couverture prédictive, l’apprentissage continu et la découverte active. Ils incluent des limites, des hypothèses réfutées et des résultats non concluants. Ils ne constituent pas une comparaison générale avec les LLM ou les modèles du monde.

## What is being explored — les mécanismes

- Composition : `core/transformations.py`, `core/invariants.py`, `core/transform_graph.py`.
- État du monde : `core/world.py`, `core/model_state.py`.
- Raisonnement et planification : `core/reasoning.py`, `core/planner.py`.
- Langage ancré : `core/language_core.py`, `core/language_bridge.py`, `core/grounded.py`.
- Mesure et observation : `core/measurement.py`, `core/source_reliability.py`.
- Certificats et ancrage : `core/generative_certificate.py`, `core/truth_anchor.py`.
- Plasticité et rétention : `core/plasticity.py`, `core/safe_plasticity.py`, `core/continuum.py`.
- Découverte structurelle : `core/active_structure.py`, `core/active_world.py`.
- Branches et instantanés : `core/branching.py`.

La présence de ces modules ne prouve ni leur intégration en un modèle unique ni la confirmation de leurs hypothèses. Chaque résultat doit être relié à son protocole et à son budget.

## Reproduction

Le moteur symbolique se lance avec `python -m app.repl` et se vérifie avec `python -m unittest tests.test_wce`. Les expériences neuronales nécessitent PyTorch, leurs fixtures et leurs scripts propres. Une vérification du moteur ne reproduit pas ces expériences.

Consigner version du code, versions des dépendances, plateforme, précision numérique, graines, données, budget et empreintes des artefacts. Deux divergences numériques inter-plateformes restent connues dans la suite scientifique ; elles ne sont pas masquées par des changements de critères.

## What WCE aims to become

Un hub ouvert pour de nouvelles architectures d’IA : code scientifique complet, protocoles inspectables, environnements reproductibles, contributions, forks et comparaisons expérimentales. Les interfaces d’entraînement communes, la licence et le processus public de contribution restent à formaliser. Aucune supériorité sur les architectures dominantes n’est démontrée ici.

**Open weights are not enough. Open the architecture.**

# Architecture WCE — état du prototype

Ce document distingue les fonctions opérationnelles, les modules de recherche et les objectifs. Il décrit le code présent dans ce dépôt au 3 octobre 2026. WCE n'est pas encore un modèle frontier entraînable de bout en bout.

## What exists now

### Parcours web et moteur symbolique

```text
Navigateur → app/static/app.js → POST /api/command → app/web.py
                                                 → app/session.py
                                                 → language/command.py
                                                 → core/world.py + core/executor.py
                                                 → physics/engine.py + physics/laws.py
                                                 → language/answer.py → réponse
```

- `app/web.py` sert la page, l'API, le contrôle d'origine et la santé. Une session indépendante par cookie reste en mémoire.
- `app/session.py` instancie un monde avec une balle et un support, applique les commandes et fait progresser la simulation.
- `language/` analyse les questions et actions ; `core/` représente les entités, relations, contraintes et transformations ; `physics/` applique les lois et interactions.
- `app/repl.py` donne une autre interface au même moteur.
- `tests/` contient les tests du moteur, des modules scientifiques et de l'interface HTTP.
- Les résultats JSON et les hypothèses préenregistrées accessibles depuis la page proviennent de `results/` et `docs/`. Ils sont spécifiques à leurs protocoles ; certains verdicts sont réfutés ou non concluants.

Le service web est volontairement léger et n'importe pas PyTorch. Le code de recherche utilisant PyTorch reste dans `core/` et ses tests.

### Déploiement local

`Dockerfile` crée une image Python 3.12 avec un utilisateur non root. `compose.yaml` publie seulement le port web réservé sur l'interface locale du Dell. Le service n'utilise ni base de données, ni worker, ni proxy. L'état des sessions n'est pas durable : il recommence après un redémarrage.

## What is being explored

Le dépôt comporte des modules et expériences autour de :

- **Apprentissage structurel** : `core/active_structure.py`, `core/boundary_program.py`, `core/program_induction_core.py`.
- **Plasticité et mémoire** : `core/plasticity.py`, `core/safe_plasticity.py`, `core/symbolic_lifelong.py`, `core/continuum.py`.
- **Mesure et observation** : `core/measurement.py`, `core/source_reliability.py`, `core/epistemic_benchmark.py`.
- **Raisonnement et langage ancré** : `core/grounded.py`, `core/discourse.py`, `core/language_bridge.py`.
- **État du monde et planification** : `core/world.py`, `core/unified_agent.py`, `core/active_world.py`.
- **Évaluation** : scripts sous `scripts/`, suites `tests/`, protocoles `docs/hypotheses_*.md`, résultats `results/`.

Leur présence dans le dépôt n'implique pas qu'ils forment déjà un système intégré ni que les hypothèses associées soient confirmées. Les fichiers de résultats contiennent les mesures détaillées et les verdicts.

## What WCE aims to become

L'objectif est un hub ouvert pour architectures d'IA frontier : code source complet, entraînement reproductible sur sa propre machine, poids et configurations publiables, environnements Docker, contributions communautaires, forks, nouveaux modèles et comparaisons expérimentales.

Pour atteindre cet objectif, il reste notamment à intégrer les modules en un parcours d'entraînement commun, définir des interfaces et formats stables, publier des jeux d'évaluation reproductibles, préciser la licence et établir un processus public de contribution. Les résultats actuels ne prouvent pas une supériorité sur les LLM ou sur d'autres modèles du monde.

**Open weights are not enough. Open the architecture.**

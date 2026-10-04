# Validated Transformation Principle

Statut : principe d'architecture **soutenu expérimentalement pour l'ajustement paramétrique uniquement**
(exploratoire, partition dev). Il n'est pas appliqué automatiquement aux autres modules de WCE.

## Énoncé

Toute transformation interne susceptible de modifier un état suit la séquence :

```
State → Transformation Proposal → Evaluation → Accept / Reject → Commit / Rollback
```

Une transformation ne doit pas pouvoir détruire silencieusement un état déjà meilleur selon les critères disponibles.

Trois obligations en découlent.

1. **Conservation.** L'état initial reste disponible, intact, jusqu'à la décision. Le rollback rend cet état ; il ne le
   reconstruit pas.
2. **Évaluation observable.** L'acceptation repose sur un critère calculable au moment de la décision, sans vérité
   cachée, avec une règle déterministe pour l'égalité, la tolérance numérique et les valeurs non finies.
3. **Auditabilité.** Chaque décision laisse une trace : état initial, proposition, valeur du critère avant et après,
   action, raison.

## Ce qui est implémenté

| Élément | Emplacement | Rôle |
|---|---|---|
| `validated_fit`, `FitTransaction`, `decide` | `core/deep_fit.py` | ajustement transactionnel : propose → évalue → commit / rollback |
| `conditional_fit`, `OptimizationDecision`, `need_for_optimization` | `core/deep_optgate.py` | porte conditionnelle : faut-il seulement proposer une transformation ? |
| `OptTrace` | `core/deep_optgate.py` | état initial, meilleur état rencontré, état final |
| paramètre `validated` de `DeepSearch` | `core/deep_search.py` | active la politique à l'étape d'ajustement, et nulle part ailleurs |

Règle d'acceptation actuelle : la vraisemblance profilée des prédictions dures sur les observations doit augmenter de
plus de la tolérance (1e-9 nat, ou le gain minimal de la porte). Égalité : rollback. Proposition non finie : rollback.
État initial non fini et proposition finie : commit.

## Ce que les expériences soutiennent

Toutes en dev, vrais atomes fournis sauf mention ; détails dans `docs/rapport_validated_optimizer.md` et
`docs/rapport_objective_alignment.md`.

- Sans validation, l'ajustement détruit près de neuf bonnes solutions sur dix.
- Avec validation sur 128 tâches jamais vues : récupération au seuil 0,99 de 7 % à 54 % (+47 points, IC 95 %
  [38 ; 56]), sans coût de calcul supplémentaire.
- Le choix du critère est décisif : le même mécanisme, avec l'objectif que l'optimiseur maximise comme critère, accepte
  95 % des propositions et ne restaure que 9 points.
- Décider de ne pas transformer est une action à part entière : une porte qui préserve les candidats dont le résidu
  ressemble à du bruit conserve 98 % des bonnes solutions, contre 71 % pour la validation seule.

## Ce que le principe ne garantit pas

- Il protège contre la dégradation **selon le critère choisi**. Si ce critère est mal aligné avec ce qui compte, une
  dégradation réelle peut être acceptée (cas mesurés : un état mieux ajusté au bruit des observations, fonctionnellement
  moins bon).
- Il ne rend pas une transformation utile ; il empêche seulement qu'elle nuise sans être vue.
- Il n'a été évalué ni en test confirmatoire, ni sur le lot scellé.

## Extension possible (non réalisée)

Par proximité avec ce qui est mesuré :

1. la simplification de structure (elle retire un atome nécessaire dans environ 1,6 % des tâches) ;
2. la réinjection des atomes polis dans la synthèse ;
3. la promotion de fragments et de primitives dans les bibliothèques ;
4. les mises à jour de croyance après une mesure active.

Chaque extension exigerait son propre critère observable et sa propre expérience contrôlée à deux bras. Aucune n'est
faite : appliquer le principe partout avant de l'avoir mesuré module par module reviendrait à supposer le résultat.

## Direction : WCE auto-évolutif

Implication architecturale, **pas capacité démontrée**. Une future modification par le système de ses propres
paramètres, primitives, structures, poids, stratégies ou code devrait suivre :

```
Proposal → Evaluation → Accept / Reject → Commit / Rollback
```

La raison est la même que pour l'ajustement : un système qui se transforme a besoin de ne pas perdre un état meilleur.
Deux leçons de ce jalon se transposeraient : conserver l'état avant de le transformer, et choisir un critère
d'acceptation qui mesure réellement ce qu'on veut préserver. Plus la transformation est profonde (structure, code),
plus le critère est difficile à définir ; c'est le problème ouvert, et rien ici ne le résout.

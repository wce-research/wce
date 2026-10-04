# Rapport — WCE Objective Alignment & Conditional Optimization

Sous-jalon EXPLORATOIRE du jalon « Deep Compositional Structure Discovery & Model-Class Boundary ».
Données : partition **dev uniquement**. Ni la validation, ni le test confirmatoire, ni le lot scellé n'ont été touchés.
Tout est `POST-HOC / EXPLORATORY`.

Sources : `results/deep_objective_summary.json` ; runs AWS `wce-deepobj-20261004-165859` (étude par candidat),
`wce-deepe2e-20261004-170452` (chaîne complète), `wce-deepresid-20261004-172752` (dégradations résiduelles), aucune
unité en erreur. Cibles : `make deep-run-objectives`, `deep-tune-gate`, `deep-run-e2e`, `deep-run-residual`,
`deep-obj-aggregate`, `deep-obj-report`.

Jeux de tâches, tous en dev et à structures disjointes entre eux et des jeux précédents : **réglage** de la porte (64
tâches, 448 candidats) ; **évaluation** (128 tâches, 896 candidats). Chaque jeu de 128 tâches compte 64 structures
observées dans deux régimes.

## Résumé

- **Quand optimiser** : quand le candidat est approximatif. Le bénéfice est nul ou négatif pour une erreur de paramètres
  d'environ 0,07, positif à partir d'environ 0,17, et fort au-delà.
- **Quand ne rien faire** : quand le candidat explique déjà les observations aussi bien que le permet le bruit.
- **Aucun des six objectifs testés n'est réellement aligné** avec la qualité fonctionnelle sur des candidats déjà bons.
- **La porte conditionnelle** est la seule politique proche du meilleur dans les trois régimes d'entrée au niveau des
  candidats. Elle préserve 98 % des bonnes solutions.
- **Avec des primitives découvertes, le gain ne survit pas** : garde-fou et porte ne font pas mieux, de façon
  significative, que « toujours optimiser ». Les candidats y sont presque tous mauvais ; il n'y a rien à préserver.

## Les trois régimes d'entrée

| Régime | Candidats (évaluation) | Accord initial moyen | Candidats déjà ≥ 0,99 |
|---|---|---|---|
| A — vrais atomes, vrais paramètres | 128 | 1,000 | 100 % |
| B — vrais atomes, perturbation faible (σ = 0,03) | 128 | 0,972 | 5 % |
| B — perturbation moyenne (σ = 0,10) | 128 | 0,924 | 0 % |
| B — perturbation forte (σ = 0,30) | 128 | 0,804 | 0 % |
| C — atomes découverts, 3 premiers candidats de la synthèse | 384 | 0,819 | 0 % |

## 1. Quand l'optimisation est-elle réellement utile ?

Courbe « erreur initiale de paramètres → bénéfice attendu de l'optimisation » (variation d'accord, objectif actuel) :

| Erreur initiale (écart maximal de paramètre) | Accord initial | Toujours optimiser | Garde-fou | Porte conditionnelle |
|---|---|---|---|---|
| 0,00 | 1,000 | −0,033 | −0,009 | −0,001 |
| 0,07 | 0,972 | −0,005 | −0,003 | +0,001 |
| 0,17 | 0,924 | +0,043 | +0,043 | +0,039 |
| 0,50 | 0,804 | +0,156 | +0,156 | +0,156 |

Le bénéfice change de signe entre 0,07 et 0,17. Avec des atomes découverts (régime C, accord initial 0,819),
l'optimisation rapporte +0,019 d'accord par candidat : 73 % des candidats s'améliorent, 17 % se dégradent.

En chaîne complète avec primitives découvertes, ne jamais optimiser coûte 0,045 d'accord (IC 95 % [0,031 ; 0,058]) et
fait tomber la récupération à 0,97 de 16 % à 3 %. **Oui : avec des paramètres approximatifs, l'optimisation est utile.**

## 2. Quand faut-il ne rien faire ?

Quand le candidat est déjà bon. En régime A, optimiser détruit 90 à 93 % des bonnes solutions selon l'objectif ; ne rien
faire les préserve toutes. En chaîne complète avec les vrais atomes (128 tâches) :

| Politique | Récupération ≥ 0,99 | ≥ 0,97 | Accord moyen | Équivalents-ajustement | Temps |
|---|---|---|---|---|---|
| Toujours optimiser | 6,3 % | 58,6 % | 0,963 | 28,3 | 5,6 s |
| Garde-fou | 57,0 % | 76,6 % | 0,980 | 27,4 | 4,6 s |
| Porte conditionnelle | 72,7 % | 79,7 % | 0,980 | 19,8 | 2,5 s |
| Ne jamais optimiser | 88,3 % | 89,1 % | 0,988 | 16,0 | 1,0 s |

Signaux observables qui distinguent les cas (calculés avant tout ajustement, moyenne sur les candidats d'évaluation) :

| Issue réelle de l'optimisation | Candidats | Taux d'erreur initial | Regroupement spatial du résidu (z) | Écart à la référence locale (z) | Gradient par point |
|---|---|---|---|---|---|
| A aidé | 567 | 0,18 | 5,7 | +2,6 | 0,44 |
| Sans effet | 167 | 0,12 | 2,8 | −1,4 | 0,19 |
| A nui | 162 | 0,14 | 3,1 | −0,8 | 0,25 |

L'écart à une référence non paramétrique locale (vote de 7 voisins) est le signal le plus net : positif quand
l'optimisation aide, négatif quand elle nuit ou ne sert à rien. Le « désaccord entre critères » mesuré par un pas
d'essai ne sépare pas les cas.

## 3. Quel objectif est le mieux aligné avec la qualité réelle ?

Aucun ne l'est sur des candidats déjà bons. Même mécanisme pour tous les objectifs différentiables (Adam, 100 pas) ;
l'erreur discrète est optimisée par recherche directe.

**Régime A (accord initial 1,000)** — où un objectif aligné ne devrait rien abîmer :

| Objectif | Accord final | Candidats dégradés | L'objectif s'améliore ET l'accord baisse | Corrélation objectif ↔ accord | ≥ 0,99 conservés |
|---|---|---|---|---|---|
| Substitution actuel | 0,968 | 100 % | 89 % | 0,36 | 7 % |
| Vraisemblance profilée adoucie | 0,955 | 100 % | 84 % | 0,49 | 9 % |
| Erreur discrète observable | 0,993 | 75 % | 75 % | −0,24 | 72 % |
| Vraisemblance + a priori sur le déplacement | 0,981 | 99 % | 98 % | 0,27 | 19 % |
| Local autour des frontières | 0,966 | 100 % | 88 % | 0,43 | 8 % |
| Robuste aux outliers | 0,967 | 100 % | 90 % | 0,40 | 10 % |

**Autres régimes** (accord final ; part où l'objectif s'améliore et l'accord baisse) :

| Objectif | B faible (0,972) | B moyen (0,924) | B fort (0,804) | C découverts (0,819) |
|---|---|---|---|---|
| Substitution actuel | 0,967 ; 43 % | 0,967 ; 6 % | 0,960 ; 2 % | 0,838 ; 15 % |
| Vraisemblance profilée adoucie | 0,958 ; 53 % | 0,955 ; 15 % | 0,947 ; 2 % | 0,837 ; 17 % |
| Erreur discrète observable | 0,977 ; 26 % | 0,967 ; 2 % | 0,894 ; 1 % | 0,824 ; 17 % |
| Vraisemblance + a priori sur le déplacement | 0,981 ; 16 % | 0,980 ; 0 % | 0,949 ; 0 % | 0,829 ; 10 % |
| Local autour des frontières | 0,966 ; 44 % | 0,965 ; 9 % | 0,953 ; 2 % | 0,837 ; 14 % |
| Robuste aux outliers | 0,967 ; 43 % | 0,966 ; 6 % | 0,959 ; 2 % | 0,839 ; 15 % |

Lecture :

- **Erreur discrète** : la moins destructrice sur un candidat déjà bon (72 % conservés), mais elle dégrade encore trois
  candidats sur quatre — réduire les erreurs sur des observations bruitées, c'est s'ajuster au bruit — et elle est la
  plus faible sur les candidats très perturbés ou découverts.
- **A priori sur le déplacement** : le mieux aligné pour des perturbations faibles et moyennes (l'accord ne baisse
  jamais quand l'objectif monte en perturbation moyenne), mais moins bon que l'objectif actuel sur les atomes
  découverts.
- **Objectif actuel, local, robuste** : équivalents entre eux à moins de 0,01 près ; les meilleurs sur les candidats très
  perturbés et découverts.
- **Vraisemblance profilée adoucie** : jamais la meilleure.

Aucun objectif ne domine dans tous les régimes. Le réglage sur le jeu de réglage retient l'objectif robuste, à égalité
pratique avec l'objectif actuel.

## 4. Le gate conditionnel bat-il always / never optimize ?

Au niveau des candidats, oui. Porte réglée sur le jeu de réglage uniquement, puis évaluée sur le jeu d'évaluation :
optimiser si le résidu est spatialement structuré (z ≥ 4) ou si le candidat n'est pas nettement meilleur que la
référence locale (z ≥ −1) ; sinon préserver ; une proposition n'est acceptée que si elle gagne au moins 2 nats de
vraisemblance profilée ; l'état retenu est le meilleur état rencontré.

Les quatre taux demandés (896 candidats, 134 bons au départ, 762 mauvais ; objectif robuste retenu par le réglage —
les chiffres avec l'objectif actuel sont identiques à 0,001 près sur l'accord moyen) :

| Politique | Bonnes solutions préservées | Bonnes détruites | Mauvaises améliorées | Mauvaises non réparées | Mauvaises empirées | Accord moyen | ≥ 0,99 | Candidats optimisés |
|---|---|---|---|---|---|---|---|---|
| Toujours optimiser | 10 % | 90 % | 76 % | 24 % | 16 % | 0,911 | 6,0 % | 100 % |
| Ne jamais optimiser | 100 % | 0 % | 0 % | 100 % | 0 % | 0,880 | 15,0 % | 0 % |
| Garde-fou | 71 % | 29 % | 73 % | 27 % | 14 % | 0,914 | 15,1 % | 100 % |
| Garde-fou sur le meilleur état rencontré | 66 % | 34 % | 74 % | 26 % | 15 % | 0,914 | 14,3 % | 100 % |
| Porte conditionnelle | **98,5 %** | 1,5 % | 64 % | 36 % | 9 % | **0,916** | **18,1 %** | 72 % |

Par régime (accord moyen) :

| Politique | A | B faible | B moyen | B fort | C |
|---|---|---|---|---|---|
| Toujours optimiser | 0,967 | 0,967 | 0,966 | 0,959 | 0,839 |
| Ne jamais optimiser | 1,000 | 0,972 | 0,924 | 0,804 | 0,819 |
| Garde-fou | 0,990 | 0,970 | 0,966 | 0,959 | 0,839 |
| Porte conditionnelle | 0,999 | 0,973 | 0,963 | 0,958 | 0,838 |

Décisions de la porte : elle préserve 121 candidats sur 128 en régime A, 89 sur 128 en perturbation faible, 21 sur 128 en
perturbation moyenne, 0 sur 128 en perturbation forte et 17 sur 384 en régime C. Elle apprend donc bien à distinguer
besoin et danger sur les régimes contrôlés.

Son coût : elle répare moins de mauvais candidats (64 % contre 76 %), surtout en perturbation moyenne où elle laisse
passer 16 % de candidats qui auraient gagné à être optimisés (accord 0,963 contre 0,966).

« Garder le meilleur état rencontré » n'apporte rien seul (0,914 dans les deux cas) : le meilleur état selon la
vraisemblance profilée est souvent un état mieux ajusté au bruit.

Aucun composant appris n'a été introduit : la règle à deux seuils suffit à séparer les régimes contrôlés.

## 5. Les 4,7 % de dégradations restantes sont-elles expliquées ?

Oui. Les 9 tâches dégradées par le garde-fou (6 nouvelles, soit 4,7 % ; 3 du diagnostic, soit 2,3 %) ont été rejouées
avec les deux bras tracés.

| Explication | Tâches | Ce qui se passe |
|---|---|---|
| Le chemin de recherche a changé la structure | 5 | La solution du garde-fou est MOINS vraisemblable sur les données et moins bien notée que celle du pipeline gelé, et de structure différente (souvent plus petite : 2 à 4 atomes contre 3 à 5). En annulant des ajustements, le garde-fou réinjecte d'autres atomes dans la synthèse, qui ne produit plus la structure que l'autre bras trouvait. Le critère d'acceptation n'est pas en cause. |
| Le bruit est exploité par un ajustement accepté | 2 | Un ajustement accepté améliore la vraisemblance profilée et dégrade l'accord réel ; la solution finale du garde-fou explique mieux les données (par exemple −115,9 contre −117,8) et est fonctionnellement moins bonne (0,841 contre 0,951). |
| Les données préfèrent la moins bonne fonction | 2 | La solution du garde-fou a un meilleur score (pénalité de complexité comprise) ou une meilleure vraisemblance, et un accord plus faible. Le score profilé est insuffisant. |

Détail utile : dans ces 9 tâches, 42 des 69 ajustements acceptés par le garde-fou font baisser l'accord réel. Accepter
un ajustement parce qu'il explique mieux des observations bruitées reste donc risqué ; c'est la raison du gain minimal
de 2 nats retenu pour la porte. Dans 7 des 9 tâches, aucun des deux bras ne retient la vraie forme ; les écarts d'accord
vont de 0,001 à 0,110.

## 6. Le gain tient-il avec primitives découvertes ?

**Non.** Chaîne complète `observations → découverte de primitives → composition → politique d'optimisation →
sélection`, 128 tâches d'évaluation, primitives découvertes par la méthode de référence (ajustement par gradient
multi-initialisations) :

| Politique | ≥ 0,90 | ≥ 0,95 | ≥ 0,97 | ≥ 0,99 | Accord moyen | Différence d'accord avec « toujours » | IC 95 % | Améliorées / dégradées |
|---|---|---|---|---|---|---|---|---|
| Toujours optimiser | 53,9 % | 28,1 % | 16,4 % | 5,5 % | 0,876 | — | — | — |
| Ne jamais optimiser | 32,0 % | 9,4 % | 3,1 % | 0,0 % | 0,831 | −0,045 | [−0,058 ; −0,031] | 19 % / 76 % |
| Garde-fou | 53,9 % | 28,9 % | 17,2 % | 5,5 % | 0,877 | +0,001 | [−0,002 ; +0,004] | 9 % / 5 % |
| Porte conditionnelle | 52,3 % | 22,7 % | 14,1 % | 4,7 % | 0,881 | +0,005 | [−0,003 ; +0,013] | 52 % / 41 % |

Ni le garde-fou ni la porte ne se distinguent de « toujours optimiser » : les intervalles incluent zéro et la
récupération à 0,99 ne bouge pas. La porte améliore légèrement l'accord moyen mais fait reculer les seuils 0,95 et 0,97
(−5,5 et −2,3 points, non significatifs).

Par profondeur, porte contre garde-fou (différence d'accord) : −0,006 à d2, −0,009 à d3, **+0,017 [+0,000 ; +0,033]** à
d4, **+0,015 [+0,001 ; +0,029]** à d5. Un léger avantage aux profondeurs 4 et 5, à la limite de la signification, sur
32 tâches par profondeur.

Par régime : standard +0,008 [−0,005 ; +0,021] ; riche +0,002 [−0,008 ; +0,013] pour la porte contre « toujours ».

Pourquoi le gain ne survit pas : avec des primitives découvertes, les candidats ont un accord initial de 0,82 et aucun
n'atteint 0,99. Le garde-fou et la porte protègent les bonnes solutions ; il n'y en a pas à protéger. Le facteur
limitant redevient la découverte de primitives.

## 7. L'optimisation devient-elle utile quand les paramètres sont approximatifs ?

Oui (section 1) : +0,043 d'accord pour une perturbation moyenne, +0,156 pour une perturbation forte, +0,019 par candidat
avec des atomes découverts, et +0,045 en chaîne complète par rapport à ne pas optimiser. Dans ces régimes,
l'optimisation répare 92 à 98 % des candidats perturbés et 73 % des candidats découverts.

Elle ne suffit pas pour autant : après optimisation, 8 à 9 % seulement des candidats perturbés atteignent 0,99, et 2 %
des candidats découverts. L'optimisation rapproche, elle ne finit pas le travail.

## 8. Quelle règle doit devenir l'implémentation WCE de référence ?

**La porte conditionnelle explicite, suivie d'une validation**, avec l'objectif actuel inchangé :

```
candidat → signaux observables → résidu structuré (z ≥ 4) ou candidat pas meilleur que la référence locale (z ≥ −1) ?
   non → PRESERVE
   oui → proposition de l'optimiseur → meilleur état rencontré → gain de vraisemblance profilée ≥ 2 nats ?
            oui → commit        non → rollback
```

Justification par les données :

- c'est la seule politique proche du meilleur dans les trois régimes au niveau des candidats (0,999 / 0,958 à 0,973 /
  0,838), quand « toujours » échoue en A et « jamais » échoue en B et C ;
- avec les vrais atomes elle récupère 72,7 % à 0,99 contre 57,0 % pour le garde-fou seul, pour 28 % d'équivalents-ajustement en moins ;
- avec des primitives découvertes elle ne fait pas moins bien que « toujours optimiser » ;
- elle est interprétable : deux seuils sur deux signaux, une marge d'acceptation, et chaque décision est tracée
  (`OptimizationDecision`).

Réserves, à trancher à la revue :

- avec les vrais atomes, « ne jamais optimiser » reste meilleur (88,3 %). La porte est un compromis, pas un optimum par
  régime ;
- son avantage en chaîne complète avec primitives découvertes n'est pas établi ;
- les seuils ont été réglés sur 64 tâches de dev ; ils ne sont validés ni sur la validation ni en confirmatoire ;
- changer d'objectif n'est pas justifié : aucun n'est aligné, et l'a priori sur le déplacement, le plus prometteur pour
  les petites perturbations, fait moins bien sur les atomes découverts.

La porte n'est pas activée par défaut : `DeepSearch(validated="gate")` l'active à l'étape d'ajustement, et nulle part
ailleurs. Le pipeline par défaut reste le pipeline gelé.

## Validated Transformation Principle

Formalisé dans `docs/architecture_validated_transformation.md`, comme principe soutenu expérimentalement pour
l'ajustement paramétrique uniquement. Ce sous-jalon y ajoute un élément : décider de ne pas transformer est une action
à part entière, et c'est elle qui fait passer la préservation de 71 % à 98,5 %. Le lien avec un WCE auto-évolutif y est
présenté comme une direction architecturale, pas comme une capacité démontrée.

## Limites

- Tout est en dev, exploratoire ; 64 structures par jeu de 128 tâches.
- La porte et le garde-fou ne font rien pour la découverte de primitives, qui reste le goulot : en chaîne complète,
  5 % de récupération à 0,99 quelle que soit la politique.
- Le régime C n'utilise qu'une méthode de découverte (gradient multi-initialisations) et les trois premiers candidats de
  la synthèse.
- Les perturbations du régime B sont gaussiennes et indépendantes par paramètre ; des erreurs réelles de découverte
  peuvent être structurées autrement.
- L'a priori sur le déplacement (σ = 0,1) et la contamination robuste (γ = 0,2) ont été testés à une seule valeur.
- Le signal « stabilité » n'a pas été calculé ; « confiance » est mesurée par l'écart à la référence locale, et
  « distance entre hypothèses » n'est pas un signal de la porte.
- Dans 5 des 9 dégradations résiduelles, la cause est un effet de chemin dans la synthèse, que ni le garde-fou ni la
  porte ne traitent.

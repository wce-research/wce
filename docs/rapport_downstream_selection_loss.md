# Rapport diagnostique — WCE Downstream Selection Loss

Sous-jalon EXPLORATOIRE du jalon « Deep Compositional Structure Discovery & Model-Class Boundary ».
Données : partition **dev uniquement**, 128 tâches (profondeurs 2 à 5 × régimes standard et riche × 16 tâches), **vrais
atomes fournis** comme littéraux : la découverte de primitives est entièrement neutralisée. Ni la validation, ni le
test confirmatoire, ni le lot scellé n'ont été touchés. Tout est `POST-HOC / EXPLORATORY`.

Source : `results/deep_downstream_summary.json` (run AWS `wce-deepdown-20261004-153523`, aucune unité en erreur).
Cibles : `make deep-run-down`, `make deep-down-aggregate`, `make deep-down-report`.

Méthode : le pipeline n'a PAS été modifié. Une sous-classe d'instrumentation (`TracedSearch`) enregistre chaque
ajustement (programme avant, programme après) et la simplification en appelant le code existant ; des tests vérifient
qu'elle produit exactement la même sélection, les mêmes scores et le même coût. Tout ce qui utilise la vérité est
calculé après coup.

## Résumé

Une solution presque parfaite est trouvée dans 99 % des tâches, puis **détruite par le premier ajustement par gradient**.

| | Part des tâches avec un candidat d'accord ≥ 0,99 |
|---|---|
| Après la génération (avant tout ajustement) | 99 % |
| Après l'ajustement initial | 13 % |
| À la sélection finale | 9 % |

Cause principale des 116 échecs : `INITIAL_FIT_BAD` dans 102 cas (88 %). La simplification, le classement, le score et
la canonicalisation sont des causes marginales (5 cas à eux tous). Le problème est d'**optimisation**, pas de sélection
ni de métrique.

## 1. Une bonne solution est-elle trouvée avant la fin ?

Oui, presque toujours. La synthèse sur signatures, avec les vrais atomes, produit un candidat d'accord ≥ 0,99 dans
127 tâches sur 128 (99,2 %). Dans 90 % de ces cas, ce candidat est aussi le premier au score avant ajustement. Le
candidat classé premier avant ajustement a un accord moyen de 0,9956 et atteint 0,99 dans 89 % des tâches.

Le système a donc déjà en main la bonne solution, avec les bons paramètres, avant de commencer à ajuster.

## 2. À quelle étape est-elle perdue ?

À l'ajustement initial. Part des tâches où un candidat d'accord ≥ 0,99 existe, étape par étape :

| Cellule | Vrais atomes | Génération | Ajustement initial | Réajustement | Simplification | Canonicalisation / dédoublonnage | Score | Classement | Sélection finale |
|---|---|---|---|---|---|---|---|---|---|
| Standard d2 | 1,00 | 1,00 | 0,12 | 0,12 | 0,06 | 0,06 | 0,06 | 0,06 | 0,06 |
| Standard d3 | 1,00 | 1,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| Standard d4 | 1,00 | 0,94 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| Standard d5 | 1,00 | 1,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| Riche d2 | 1,00 | 1,00 | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 | 0,50 |
| Riche d3 | 1,00 | 1,00 | 0,38 | 0,38 | 0,19 | 0,31 | 0,19 | 0,19 | 0,19 |
| Riche d4 | 1,00 | 1,00 | 0,06 | 0,06 | 0,00 | 0,06 | 0,00 | 0,00 | 0,00 |
| Riche d5 | 1,00 | 1,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |

Étape où la bonne solution est perdue (`BestCandidateTrajectory`), sur les 128 tâches : ajustement initial 110,
simplification 5, génération 1, jamais perdue 12. Les colonnes « réajustement », « canonicalisation » mesurent le
meilleur candidat encore disponible ; les colonnes « simplification », « score », « classement » et « sélection finale »
mesurent le candidat retenu à cette étape.

Meilleur accord moyen par étape : 1,000 à la génération ; 0,933 à 0,986 après l'ajustement initial ; 0,898 à 0,982 à la
sélection finale.

## 3. Quelle étape contribue le plus à la dégradation ?

L'ajustement par gradient (polissage) des candidats.

| Cause principale | Standard | Riche | Total |
|---|---|---|---|
| `INITIAL_FIT_BAD` | 55 | 47 | 102 |
| Récupérée (accord final ≥ 0,99) | 1 | 11 | 12 |
| `THRESHOLD_TOO_STRICT` | 6 | 2 | 8 |
| `SCORING_MISALIGNED` | 0 | 2 | 2 |
| `CANONICALIZATION_CHANGED_BEHAVIOR` (dédoublonnage) | 1 | 1 | 2 |
| `RANKING_FAILURE` | 0 | 1 | 1 |
| `OTHER` (aucun candidat ≥ 0,99 généré) | 1 | 0 | 1 |
| `POLISH_DEGRADED`, `SIMPLIFICATION_DEGRADED`, `FINAL_SELECTION_FAILURE`, `NUMERICAL_INSTABILITY`, `FUNCTIONAL_EQUIVALENCE_MISCOUNTED` | 0 | 0 | 0 |

Par profondeur, `INITIAL_FIT_BAD` : standard 12, 12, 15, 16 sur 16 (d2 à d5) ; riche 7, 9, 15, 16 sur 16.

Définitions utilisées : `INITIAL_FIT_BAD` = un candidat ≥ 0,99 a été soumis à l'ajustement et plus aucun candidat ≥ 0,99
n'existe après ; `THRESHOLD_TOO_STRICT` = même situation, mais l'ajustement a amélioré la vraisemblance sur les données
et l'accord final reste ≥ 0,97 (la perte est statistique, pas un défaut d'optimisation). `SIMPLIFICATION_DEGRADED`
n'apparaît jamais comme cause principale parce que, lorsque la simplification dégrade, le candidat avait déjà été abîmé
en amont.

## 4. Le score interne est-il mal aligné avec la qualité fonctionnelle ?

Modérément, et ce n'est pas la cause dominante. Classement oracle (par accord fonctionnel réel) contre scores internes,
sur toutes les structures ajustées de chaque tâche :

| Score | Corrélation de rang | Accord du premier (top-1) | Meilleur dans les 3 premiers | … dans les 5 premiers | Accord du candidat classé premier |
|---|---|---|---|---|---|
| Score a posteriori (actuel) | 0,78 | 0,64 | 0,79 | 0,85 | 0,953 |
| Vraisemblance profilée | 0,86 | 0,51 | 0,85 | 0,95 | 0,956 |
| Vraisemblance robuste adoucie | 0,85 | 0,36 | 0,78 | 0,90 | 0,955 |
| MDL | 0,77 | 0,64 | 0,79 | 0,84 | 0,953 |
| Pénalité de paramètres seule | −0,78 | 0,01 | 0,01 | 0,02 | 0,717 |
| Pénalité de structure seule | −0,74 | 0,01 | 0,01 | 0,02 | 0,717 |

Lecture : la vraisemblance est la composante la mieux corrélée à la qualité réelle ; les pénalités de complexité vont à
l'encontre de la qualité (les bonnes structures sont les plus grosses). Le score composite choisit le meilleur candidat
disponible dans 64 % des tâches. En moyenne sur les 128 tâches (écart nul quand il le choisit), le meilleur disponible a une vraisemblance plus
élevée de 1,7 nat mais paie 7,7 nats de pénalité de paramètres et 3,3 de pénalité de structure de plus : c'est la
pénalité de complexité qui fait perdre les bonnes structures, pas la vraisemblance.

Mais l'enjeu est faible ici : choisir systématiquement le meilleur candidat disponible après ajustement ne donnerait que
13 % de récupération (contre 9 %), parce que les bons candidats ont déjà été abîmés.

### Calibration du score final

La confiance interne (masse de la structure retenue) est très mal calibrée : confiance moyenne 0,905 pour un taux de
récupération réel de 0,094.

| Tranche de confiance | Tâches | Confiance moyenne | Récupération ≥ 0,99 | Récupération ≥ 0,97 |
|---|---|---|---|---|
| 0,2 – 0,4 | 1 | 0,26 | 0,00 | 0,00 |
| 0,4 – 0,6 | 12 | 0,49 | 0,00 | 0,25 |
| 0,6 – 0,8 | 8 | 0,68 | 0,13 | 0,13 |
| 0,8 – 1,0 | 107 | 0,98 | 0,10 | 0,56 |

ECE 0,81 et Brier 0,76 au seuil 0,99 ; ECE 0,41 et Brier 0,39 au seuil 0,97.

## 5. Le seuil 0,99 explique-t-il une part importante du problème ?

Une part, pas l'essentiel. Part des tâches dont l'accord final atteint chaque seuil :

| Cellule | ≥ 0,90 | ≥ 0,95 | ≥ 0,97 | ≥ 0,99 | Structure exacte | Structure équivalente | Accord final moyen |
|---|---|---|---|---|---|---|---|
| Standard d2 | 1,00 | 0,81 | 0,63 | 0,06 | 0,81 | 0,81 | 0,966 |
| Standard d3 | 0,88 | 0,81 | 0,50 | 0,00 | 0,25 | 0,56 | 0,960 |
| Standard d4 | 0,75 | 0,44 | 0,13 | 0,00 | 0,31 | 0,38 | 0,923 |
| Standard d5 | 0,56 | 0,00 | 0,00 | 0,00 | 0,06 | 0,13 | 0,898 |
| Riche d2 | 1,00 | 0,94 | 0,88 | 0,50 | 0,63 | 0,63 | 0,982 |
| Riche d3 | 1,00 | 1,00 | 0,81 | 0,19 | 0,56 | 0,88 | 0,982 |
| Riche d4 | 1,00 | 0,81 | 0,75 | 0,00 | 0,50 | 0,50 | 0,968 |
| Riche d5 | 0,81 | 0,69 | 0,31 | 0,00 | 0,06 | 0,19 | 0,945 |
| **Toutes** | 0,88 | 0,69 | 0,50 | 0,09 | | | |

41 % des tâches se situent entre 0,97 et 0,99 : l'échec au seuil 0,99 est en partie un effet de seuil. Mais la moitié
des tâches reste sous 0,97, et à la profondeur 5 en régime standard aucune n'atteint 0,95. L'échec est réel.

Le seuil n'est pas non plus inatteignable : le candidat disponible AVANT ajustement atteint 0,99 dans 89 % des tâches.
Seules 8 tâches sur 128 relèvent de `THRESHOLD_TOO_STRICT`.

Représentations équivalentes : une structure retenue fonctionnellement correcte (≥ 0,99) mais de forme non équivalente à
la vraie apparaît dans 2,3 % des tâches ; elle est comptée comme un succès fonctionnel, pas comme un échec. À l'inverse,
la structure exacte est souvent retenue avec des paramètres abîmés (81 % de structures exactes en standard d2 pour 6 % de
récupération à 0,99).

## 6. La simplification est-elle sûre ?

Presque, et elle agit rarement. Elle modifie la structure retenue dans 3 tâches sur 128 (2,3 %). Dans 2 de ces 3 cas
elle retire un atome nécessaire (accord −0,050 et −0,007) ; dans le troisième elle améliore légèrement l'accord
(+0,005). Elle ne fusionne jamais deux structures : elle remplace une structure par l'un de ses enfants, réajusté
(60 pas), donc elle modifie les paramètres quand elle agit. La règle (facteur de Bayes < 3) est donc trop permissive
dans environ 1,6 % des tâches. Elle n'est la cause principale d'aucun échec.

## 7. Le polissage est-il sûr ?

**Non.** Sur les 713 ajustements de candidats dont l'accord avant ajustement était ≥ 0,97 :

| | |
|---|---|
| Accord moyen avant → après | 0,994 → 0,956 |
| Candidats dégradés (accord −0,002 ou plus) | 95 % |
| Candidats parfaits (accord ≥ 0,999) tombant sous 0,99 | 93 % (379 sur 408) |
| Chute moyenne d'accord | 0,038 |
| Déplacement moyen du paramètre le plus modifié | 0,67 |
| Ajustements qui font BAISSER la vraisemblance sur les données | 55 % |
| Variation moyenne du nombre d'erreurs sur les données | +1,35 |

Le polissage dégrade donc silencieusement, et dans plus de la moitié des cas le résultat est objectivement moins bon sur
les données disponibles que le point de départ — sans que rien ne s'y oppose, puisque la version initiale n'est jamais
enregistrée.

Par cellule, part des candidats ≥ 0,97 rendus moins vraisemblables : standard 49 %, 28 %, 36 %, 42 % (d2 à d5) ; riche
74 %, 48 %, 72 %, 85 %. Chute moyenne d'accord : 0,031, 0,028, 0,064, 0,066 (standard) ; 0,018, 0,015, 0,039, 0,038
(riche).

Borne d'estimation : la VRAIE structure, ajustée depuis ses VRAIS paramètres, finit avec un accord moyen de 0,774 et
passe de 39 à 118 erreurs sur les données en moyenne ; elle n'améliore sa vraisemblance que dans 16 % des tâches. Ce
n'est pas un problème de données insuffisantes : l'optimiseur quitte un optimum pour un point nettement moins bon.
Explication la plus probable, non testée isolément : l'ajustement démarre avec une frontière très adoucie (raideur 4)
et un pas de 0,05, ce qui éloigne des paramètres déjà justes avant que la raideur ne remonte.

## 8. Le nouveau fitter change-t-il le ranking ?

Parfois, et l'équivalence numérique annoncée ne tient pas en général. Ancien ajusteur (`fit_robust`) contre nouvel
ajusteur par lots, mêmes candidats, même initialisation, 100 pas :

| | |
|---|---|
| Tâches où l'écart de paramètres dépasse 1e-3 | 28 % |
| Écart maximal de paramètres | 1,01 |
| Écart maximal de score | 24,5 nats |
| Écart maximal d'accord fonctionnel | 0,035 |
| Même candidat classé premier | 98,4 % |
| Même ordre complet | 86,7 % |
| Corrélation de rang moyenne | 0,985 |

L'écart de 1e-7 mesuré au début du jalon venait de cas courts et bien conditionnés. Sur des trajectoires de 100 pas
partant de candidats composites, une différence d'arrondi de l'initialisation (l'ancien ajusteur passe par des flottants
simple précision) s'amplifie : l'optimisation est numériquement sensible. Conséquence pour le pré-enregistrement :
l'hypothèse H10(b) (écart ≤ 1e-6) ne tient pas telle qu'elle est formulée sur ces tâches. Ce n'est pas la cause de la
perte aval — l'ancien ajusteur dégrade autant — mais c'est un symptôme du même défaut : l'ajustement est instable.

## 9. Le problème est-il principalement optimisation, sélection ou métrique ?

**Optimisation.**

| Famille | Tâches | Part des 116 échecs |
|---|---|---|
| Optimisation (`INITIAL_FIT_BAD`) | 102 | 88 % |
| Métrique (`THRESHOLD_TOO_STRICT`) | 8 | 7 % |
| Sélection (score, classement, dédoublonnage) | 5 | 4 % |
| Génération (`OTHER`) | 1 | 1 % |

Contrefactuels calculés sur les états enregistrés, sans modifier le pipeline :

| Scénario | ≥ 0,99 | ≥ 0,97 | Accord moyen |
|---|---|---|---|
| Pipeline actuel | 9 % | 50 % | — |
| Sans simplification | 9 % | 50 % | 0,953 |
| Sans dédoublonnage | 9 % | 50 % | 0,953 |
| Meilleur candidat disponible après ajustement (sélection oracle) | 13 % | 60 % | 0,968 |
| Garde-fou « hold-best » (voir section 10) | 57 % | 72 % | 0,975 |
| Candidat classé premier AVANT ajustement, sans ajuster | 89 % | 96 % | 0,996 |

Améliorer la sélection ne peut pas rapporter plus de 4 points ; ne pas dégrader à l'ajustement en rapporte 48 à 80.

## 10. Quelle correction minimale est justifiée par les données ?

**Un garde-fou à l'étape d'ajustement, et rien d'autre** : ne jamais remplacer un candidat par sa version ajustée si
celle-ci est moins vraisemblable sur les données disponibles.

- Il ne touche qu'un étage (l'enregistrement du résultat d'un ajustement). Score, classement, simplification,
  canonicalisation et critères d'acceptation restent inchangés.
- Il n'utilise aucune vérité : la comparaison se fait sur la vraisemblance profilée des données observées.
- Estimation contrefactuelle (exploratoire, calculée sur les états enregistrés) : récupération à 0,99 de 9 % → 57 % ;
  à 0,97 de 50 % → 72 %. Par cellule : standard 56 %, 44 %, 31 %, 13 % (d2 à d5) ; riche 81 %, 69 %, 88 %, 75 %.
- Il n'est PAS intégré au pipeline : seul le contrefactuel a été calculé.

Ce garde-fou ne récupère pas tout (57 % contre 89 % sans ajustement) : quand la version ajustée est légèrement plus
vraisemblable mais fonctionnellement moins bonne, il la garde. Ce résidu relève du surajustement au bruit, pas de
l'optimiseur.

### Expérience contrôlée proposée

1. Un seul changement : enregistrer la version initiale de chaque candidat et ne la remplacer que si l'ajustement
   augmente la vraisemblance profilée. Aucun autre étage modifié.
2. Deux bras, mêmes tâches, mêmes graines : pipeline actuel contre pipeline + garde-fou.
3. Tâches : les 128 de ce diagnostic (pour vérifier que le contrefactuel se réalise) et 128 nouvelles tâches de dev
   jamais vues (pour ne pas juger sur les tâches qui ont motivé la correction).
4. Mesures : récupération à 0,90 / 0,95 / 0,97 / 0,99, accord moyen, part des ajustements acceptés, coût.
5. Critère annoncé d'avance : le garde-fou est retenu si la récupération à 0,99 sur les nouvelles tâches dépasse celle du
   pipeline actuel d'au moins 20 points, sans baisse d'accord moyen.

Une seconde correction possible — changer le schéma de l'optimiseur (raideur initiale, pas) pour qu'il ne quitte plus un
bon point — touche un autre mécanisme. Elle ne doit être testée qu'après, et séparément, pour ne pas confondre les deux
effets.

## Limites

- 16 tâches par cellule : les proportions par cellule ont une résolution de 1/16.
- La cause « raideur initiale trop faible » du polissage est une explication plausible, pas un résultat : elle n'a pas
  été isolée expérimentalement.
- Le contrefactuel « hold-best » est calculé a posteriori ; son effet réel dans la boucle (les tours de réajustement
  réinjectent les atomes polis) peut différer.
- La colonne « canonicalisation » mêle la normalisation des clés (qui ne change jamais le comportement : 0 cas) et le
  dédoublonnage fonctionnel (qui écarte en moyenne 3,4 structures par tâche et coûte le bon candidat dans 2 tâches).
- Diagnostic conduit avec les vrais atomes : il ne dit rien de la chaîne complète avec des atomes découverts.
- Rien ici n'est confirmatoire.

# WCE End-to-End Composition with Discovered Atoms — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucune campagne 16 graines, lot scellé jamais touché.
Le pipeline gelé reste la référence officielle ; `flat_scene_k3` reste une baseline candidate ; la composition sur
classes n'est pas branchée.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur les clés de composition 990–999 (réglage de la chaîne C sur 60 tâches, clés
990–994 ; essais sur la clé 999). Évaluation sur **384 tâches jamais utilisées** : clés 900–931, profondeurs 1 à 6,
régimes standard et rich. Aucune de ces tâches n'a servi à sélectionner `flat_scene_k3`.

### Les trois chaînes, et elles seules, sont primaires

| Chaîne | Primitives | Composition | Aval |
|---|---|---|---|
| **A** | pipeline gelé (gradient) | pipeline gelé | pipeline gelé |
| **B** | `flat_scene_k3` | pipeline actuel | pipeline actuel (ajustement non validé, score actuel) |
| **C** | `flat_scene_k3` | composition sur classes d'équivalence | finalisation validée |

Aucune des trois ne reçoit de vrai type, de vrai paramètre, de vrai atome ni d'appartenance vraie. La profondeur
maximale est fixée à 6 pour toutes : la taille de la vérité n'est donnée à personne.

Configuration de C, figée avant le lancement (`results/deep_configs/e2e.json`) : primitives regroupées en hypothèses
(une primitive et ses équivalentes ou complémentaires = une hypothèse), réduction à six atomes au plus par sélection
avant sur l'erreur plancher (gain minimal 1), composition exacte sur cellules avec les compléments comme unités,
sélection par vraisemblance moins ½·ln N par feuille, ajustement **validé** (état initial conservé, commit seulement
si le critère observable progresse).

Toute autre variante (`C_nofit`, `C_forcedfit`, `C_errors`, `C_posterior`, `C_k1`, `C_k5`, autres sources, autres
réductions, moteur large, composition hiérarchique) est secondaire et ne peut pas devenir primaire après observation.

### Hypothèses

| # | Hypothèse | Critère |
|---|---|---|
| EE1 | C bat le pipeline gelé | C − A, différence appariée par tâche, IC 95 % > 0 sur la structure candidate (≥ 0,97) ET sur l'accord final |
| EE2 | C bat `flat_scene_k3` + aval actuel | C − B, mêmes deux critères |
| EE3 | le gain tient en profondeur | C − A et C − B sur l'accord final, profondeurs 3 à 6 : IC > 0 |
| EE4 | sans explosion de calcul | temps par tâche de C inférieur à celui de A et de B |
| EE5 | l'ajustement validé ne nuit pas | C − `C_nofit` ≥ 0 (IC non négatif) ; `C_forcedfit` − C ≤ 0 |
| EE6 | le score de sélection actuel reste mal aligné | C − `C_posterior` sur l'accord, IC > 0 |
| EE7 | la réduction garde les bons atomes | rappel des vrais atomes après réduction ≥ 0,9 × rappel avant |
| EE8 | le moteur résiste aux faux atomes | cas contrôlé : vrais + 3 faux, récupération à moins de 0,05 des vrais atomes seuls |
| EE9 | un atome manquant est signalé | cas contrôlé : décision autre que KNOWN dans ≥ 80 % des tâches |
| EE10 | le hors-classe est signalé | tâches hors classe : décision autre que KNOWN dans ≥ 80 % ; tâches en classe récupérées : KNOWN dans ≥ 80 % |
| EE11 | `flat_scene_k3` reste la meilleure source sans oracle | accord dans le même moteur, contre les autres sources |

Pour C, « structure candidate » désigne les 20 premières classes selon le sélecteur — pas toute la table, ce qui
serait injuste envers A et B dont le vivier ne compte qu'une vingtaine de candidats.

Attente déclarée d'après la mise au point : C atteint 0,905 d'accord sur 60 tâches de réglage, du même ordre que ce que
B obtenait sur d'autres jeux. EE1 incertaine, EE2 peu probable, EE4 probable. En mise au point, l'ajustement validé
aide quand les atomes sont découverts (+0,017), à l'inverse de ce qui était mesuré avec les vrais atomes.

Limites connues avant lancement :

- les seuils de la décision KNOWN / UNKNOWN / NONE_OF_THE_ABOVE sont des valeurs par défaut non réglées ; l'ancienne
  infrastructure de frontière de classe demande un modèle ajusté sur validation, qui n'existe pas : elle n'est pas
  réutilisée, seule une décision à deux signaux explicites est évaluée ;
- les cas contrôlés (faux atomes, atomes manquants) partent des vrais atomes : ce sont des diagnostics à oracle,
  rapportés à part.

### Règle de verdict et décision confirmatoire, fixées d'avance

- `READY FOR 16-SEED CONFIRMATORY CAMPAIGN` : EE1, EE2, EE3 et EE4 toutes soutenues. Alors seulement, proposer la
  campagne 16 graines — sans la lancer.
- `END-TO-END IMPROVED BUT INSUFFICIENT` : EE1 soutenue, mais EE2 ou EE3 non soutenue.
- `END-TO-END STILL BLOCKING` : EE1 non soutenue.

Dans les deux derniers cas : identifier le nouveau goulot et s'arrêter. Le lot scellé n'est lancé dans aucun cas.

## 1. Protocole exécuté

- Run AWS `wce-deepee-20261009-224553` (EC2 + SSM, 2 × c6i.xlarge, 71 et 73 min, 2 400 sous-unités, 0 erreur,
  13 536 lignes). Commandes : `make deep-ee-tune` (local, mise au point), `make deep-run-ee`, `make deep-fetch`,
  `make deep-ee-aggregate`, `make deep-ee-report`. Résumé : `results/deep_end_to_end_summary.json`.
- 384 tâches jamais utilisées, 64 par profondeur de 1 à 6 ; 96 tâches hors classe pour la frontière de classe.
- Code : `core/deep_e2e.py` (système, aucun oracle), `core/deep_e2e_exp.py` (évaluation), `core/deep_ee_report.py`.
  Tests : 50 nouveaux (`make deep-ee-test`) ; suite complète 5 000 tests OK, 7 ignorés ; lot scellé intact.
- Une modification au moteur du sous-jalon précédent : `CellSpace` accepte jusqu'à 12 unités quand les cellules
  occupées tiennent sur 64 bits (un atome et son complément ne créent pas de cellule). Comportement par défaut inchangé.
- Aucun réglage sur l'évaluation ; configuration de C figée avant le lancement. Une graine par tâche.
- Écart au protocole : aucun.

## 2. Comparaison principale

| Chaîne | Accord final | Hors région | Structure candidate ≥ 0,97 | Final ≥ 0,97 | Final ≥ 0,99 | Littéraux | Secondes |
|---|---|---|---|---|---|---|---|
| A — pipeline gelé | 0,880 | 0,826 | 0,31 | 0,25 | 0,13 | 83 | 23,4 |
| B — `flat_scene_k3` + aval actuel | **0,899** | **0,859** | **0,39** | **0,32** | **0,15** | 21 | 18,5 |
| C — `flat_scene_k3` + classes + finalisation validée | 0,888 | 0,845 | 0,22 | 0,23 | 0,09 | 18 → 4,6 | **1,4** |

Différences appariées par tâche, IC 95 % :

| Comparaison | Accord final | Structure candidate | Final ≥ 0,99 |
|---|---|---|---|
| C − A, toutes | +0,008 [−0,001 ; +0,015] | **−0,089 [−0,122 ; −0,055]** | −0,037 [−0,065 ; −0,010] |
| C − A, profondeurs 3–6 | +0,010 [−0,002 ; +0,021] | −0,055 [−0,090 ; −0,020] | 0,000 |
| C − B, toutes | **−0,012 [−0,020 ; −0,005]** | −0,172 [−0,208 ; −0,135] | −0,055 [−0,083 ; −0,026] |
| C − B, profondeurs 3–6 | −0,014 [−0,024 ; −0,005] | −0,137 [−0,180 ; −0,098] | 0,000 |
| B − A, toutes | +0,019 [+0,013 ; +0,026] | +0,083 [+0,050 ; +0,117] | +0,018 [0,000 ; +0,039] |
| B − A, profondeurs 3–6 | +0,024 [+0,015 ; +0,033] | +0,082 [+0,043 ; +0,125] | 0,000 |

C ne bat ni A ni B. B bat de nouveau A, sur un troisième jeu de tâches indépendant.

## 3. Par profondeur

| Profondeur | Accord A | Accord B | Accord C | Candidate A | Candidate B | Candidate C | Final ≥ 0,99 : A / B / C |
|---|---|---|---|---|---|---|---|
| 1 | 0,987 | 0,985 | 0,984 | 0,98 | 0,98 | 0,86 | 0,66 / 0,67 / 0,47 |
| 2 | 0,930 | 0,952 | 0,937 | 0,52 | 0,69 | 0,33 | 0,11 / 0,20 / 0,08 |
| 3 | 0,895 | 0,917 | 0,905 | 0,17 | 0,34 | 0,05 | 0 / 0 / 0 |
| 4 | 0,871 | 0,894 | 0,866 | 0,11 | 0,19 | 0,06 | 0 / 0 / 0 |
| 5 | 0,795 | 0,831 | 0,819 | 0,03 | 0,09 | 0,00 | 0 / 0 / 0 |
| 6 | 0,802 | 0,817 | 0,815 | 0,02 | 0,03 | 0,00 | 0 / 0 / 0 |

À partir de la profondeur 3, aucune des trois chaînes ne récupère une seule tâche à 0,99.

Rappel des candidats de la chaîne C :

| Profondeur | Vrais atomes présents dans les primitives | Présents après réduction | Faux atomes gardés | Bonne classe (≥ 0,99) générée | Dans les 20 premières | Arbre exact | Classe équivalente |
|---|---|---|---|---|---|---|---|
| 1 | 0,98 | 0,89 | 1,2 | 0,41 | 0,39 | 0,56 | 0,14 |
| 2 | 0,64 | 0,43 | 3,3 | 0,03 | 0,03 | 0,08 | 0,03 |
| 3 | 0,37 | 0,22 | 4,2 | 0,00 | 0,00 | 0,00 | 0,00 |
| 4 | 0,31 | 0,17 | 4,6 | 0,00 | 0,00 | 0,00 | 0,00 |
| 5 | 0,22 | 0,12 | 5,0 | 0,00 | 0,00 | 0,00 | 0,00 |
| 6 | 0,17 | 0,08 | 4,9 | 0,00 | 0,00 | 0,00 | 0,00 |

Un vrai atome est « présent » si une primitive le reproduit à 0,95 d'accord. À la profondeur 6, une tâche a en
moyenne un vrai atome sur six parmi ses primitives, et cinq faux atomes parmi les six gardés.

## 4. Attribution des échecs de la chaîne C

| Cause principale | Toutes (384) | Profondeurs 3–6 (256) | Profondeurs 4–6 (192) |
|---|---|---|---|
| récupérée | 35 | 0 | 0 |
| primitive correcte absente | **292** | 254 | 192 |
| bonne classe non générée | 33 | 0 | 0 |
| primitive correcte éliminée | 18 | 2 | 0 |
| ajustement dégradant | 3 | 0 | 0 |
| trop de faux atomes | 3 | 0 | 0 |
| mal classée, non identifiable, hors classe | 0 | 0 | 0 |

« Bonne classe non générée » se concentre à la profondeur 1 (24 cas) : l'atome est présent à 0,95 mais pas assez
précis pour qu'une classe atteigne 0,99 (accord 0,97 en moyenne). C'est un défaut de précision des paramètres, pas de
composition.

## 5. Variantes de la chaîne C (secondaires)

| Variante | Accord | Final ≥ 0,97 | Final ≥ 0,99 | Différence d'accord avec C, IC 95 % |
|---|---|---|---|---|
| C (ajustement validé, pénalité légère) | 0,888 | 0,23 | 0,09 | — |
| sans ajustement | 0,877 | 0,19 | 0,06 | −0,011 [−0,014 ; −0,007] |
| ajustement forcé | 0,887 | 0,24 | 0,10 | 0,000 [−0,002 ; +0,001] |
| sélection par erreurs | 0,887 | 0,22 | 0,09 | −0,001 [−0,003 ; +0,002] |
| sélection par score a posteriori existant | 0,875 | 0,25 | 0,12 | −0,012 [−0,017 ; −0,008] |
| une hypothèse par type (`k = 1`) | 0,821 | 0,20 | 0,11 | −0,067 [−0,085 ; −0,050] |
| cinq hypothèses par type (`k = 5`) | 0,896 | 0,25 | 0,08 | +0,009 [+0,003 ; +0,016] |

L'ajustement validé accepte 79 % des propositions. Le score existant perd en accord moyen mais gagne 3 points de
récupération à 0,99, tous à la profondeur 1, où préférer la plus petite structure est juste.

## 6. Sources de primitives, dans le même moteur

| Source | Oracle | Littéraux | Vrais atomes présents | Après réduction | Accord | Accord, profondeurs 4–6 |
|---|---|---|---|---|---|---|
| vrais atomes | oui | 3,5 | 1,00 | 0,87 | 0,936 | 0,898 |
| `flat_scene_k5` | non | 29 | 0,50 | 0,32 | 0,896 | 0,849 |
| `flat_scene_k3` | non | 18 | 0,45 | 0,32 | 0,888 | 0,833 |
| atomes du pipeline gelé | non | 83 | 0,38 | 0,30 | 0,888 | 0,838 |
| fragments (`hier_k3`) | non | 5,6 | 0,29 | 0,27 | 0,847 | 0,783 |
| `flat_scene_k1` | non | 6,0 | 0,33 | 0,26 | 0,821 | 0,744 |
| découverte robuste | non | 3,5 | 0,32 | 0,25 | 0,633 | 0,449 |

Contre `flat_scene_k3` : `k5` +0,009 [+0,003 ; +0,016] ; atomes du pipeline gelé 0,000 [−0,007 ; +0,008] ; fragments
−0,041 ; `k1` −0,067 ; découverte robuste −0,255 ; vrais atomes +0,049 [+0,030 ; +0,067].

## 7. Réduction et passage à l'échelle

Mêmes primitives (`flat_scene_k3`, 14 hypothèses après regroupement), aucun ajustement :

| Mécanisme | Atomes gardés | Vrais atomes gardés | Accord | Classes | Secondes |
|---|---|---|---|---|---|
| confiance (identique au dédoublonnage seul) | 6,0 | 0,31 | 0,845 | 150 000 | 0,43 |
| deux par famille | 6,0 | 0,32 | 0,850 | 151 000 | 0,44 |
| dominance | 6,0 | 0,30 | 0,852 | 196 000 | 0,52 |
| **erreur plancher (retenu)** | 3,0 | 0,30 | 0,869 | 69 000 | 0,15 |
| moteur large, 6 atomes | 6,0 | 0,31 | 0,774 | 1 100 | 0,10 |
| moteur large, 12 atomes | 11,9 | 0,38 | 0,855 | 1 400 | 0,20 |
| moteur large, toutes (14) | 14,1 | 0,40 | 0,876 | 1 400 | 0,24 |
| hiérarchique, toutes (14) | 14,1 | 0,40 | 0,878 | 2 000 | 0,39 |

Le rappel des vrais atomes est de 0,40 avant toute réduction. Contre la réduction retenue : moteur large sur toutes les
hypothèses +0,007 [+0,001 ; +0,014] ; hiérarchique +0,010 [+0,004 ; +0,017] ; confiance −0,024 [−0,031 ; −0,015].

Le passage au-delà de six atomes fonctionne : quatorze atomes en un quart de seconde, sans énumération. Il ne change
presque rien au résultat, parce que les bons atomes ne sont pas dans les primitives.

## 8. Cas contrôlés (diagnostic à oracle : on part des vrais atomes)

Chaîne C sans ajustement, primitives remplacées par :

| Condition | Accord | Final ≥ 0,99 | Atomes gardés | Faux gardés |
|---|---|---|---|---|
| vrais atomes | 0,944 | 0,79 | 2,9 | 0,0 |
| vrais + 1 faux | 0,946 | 0,77 | 3,0 | 0,1 |
| vrais + 3 faux | 0,949 | 0,77 | 3,2 | 0,3 |
| paramètres bruités (écart-type 0,10) | 0,854 | **0,02** | 2,7 | 0,7 |
| paramètres bruités (écart-type 0,25) | 0,723 | 0,00 | 2,4 | 2,0 |
| faux atomes seulement | 0,391 | 0,00 | 1,6 | 1,6 |
| un atome manquant | 0,620 | 0,01 | 1,8 | 0,0 |
| deux atomes manquants | 0,428 | 0,00 | 1,1 | 0,0 |

Contre les vrais atomes seuls : + 1 faux −0,018 [−0,034 ; −0,005] de récupération ; + 3 faux −0,018
[−0,042 ; +0,005] ; paramètres bruités à 0,10 −0,771 [−0,813 ; −0,729].

Constat à part : avec les vrais atomes et eux seuls, cette chaîne récupère 0,79 des tâches, contre 0,99 pour la
composition exhaustive du sous-jalon précédent. La différence vient de la réduction : la sélection avant sur l'erreur
plancher ne garde que 87 % des vrais atomes. Un atome qui n'agit qu'en combinaison avec un autre (ou exclusif) ne
réduit pas l'erreur plancher à lui seul et n'est jamais choisi. Récupération par profondeur : 1,00 ; 0,97 ; 0,78 ;
0,70 ; 0,70 ; 0,56.

## 9. KNOWN / UNKNOWN / NONE_OF_THE_ABOVE

| Situation | n | KNOWN | UNKNOWN | NONE_OF_THE_ABOVE |
|---|---|---|---|---|
| en classe, chaîne C, tâche récupérée | 35 | 35 | 0 | 0 |
| en classe, chaîne C, tâche NON récupérée | 349 | 337 | 9 | 3 |
| contrôlé : vrais atomes | 384 | 346 | 28 | 10 |
| contrôlé : un atome manquant | 320 | 136 | 125 | 59 |
| contrôlé : faux atomes seulement | 384 | 3 | 202 | 179 |
| hors classe : primitive absente | 32 | 32 | 0 | 0 |
| hors classe : composition absente | 32 | 28 | 4 | 0 |
| hors classe : relation absente | 32 | 29 | 3 | 0 |

La décision répond KNOWN dans 97 % des tâches en classe non récupérées et dans 93 % des tâches hors classe. Son
signal principal compare les erreurs de la structure à celles des plus proches voisins ; or une composition de primitives
approximatives explique les observations aussi bien que les voisins tout en étant fausse (écart moyen −0,036). Le
signal ne distingue que les cas grossiers : faux atomes seuls (99 % non KNOWN), un atome manquant (58 %).

## 10. Hypothèses pré-enregistrées

| # | Résultat |
|---|---|
| EE1 C bat le pipeline gelé | **non soutenue** : accord +0,008 [−0,001 ; +0,015] ; structure candidate −0,089 [−0,122 ; −0,055] |
| EE2 C bat `flat_scene_k3` + aval actuel | **non soutenue** : accord −0,012 [−0,020 ; −0,005] |
| EE3 gain en profondeur 3–6 | **non soutenue** : +0,010 [−0,002 ; +0,021] contre A ; −0,014 [−0,024 ; −0,005] contre B |
| EE4 sans explosion de calcul | **soutenue** : 1,4 s par tâche contre 23,4 et 18,5 |
| EE5 l'ajustement validé ne nuit pas | **soutenue** : +0,011 [+0,007 ; +0,014] contre l'absence d'ajustement ; l'ajustement forcé fait jeu égal |
| EE6 le score existant reste mal aligné | **soutenue** sur l'accord (+0,012 [+0,008 ; +0,017]) ; il récupère pourtant 3 points de plus à 0,99, à la profondeur 1 |
| EE7 la réduction garde les bons atomes | **non soutenue** : 0,32 après pour 0,45 avant (71 %) ; 87 % même quand tous les atomes sont vrais |
| EE8 résistance aux faux atomes | **soutenue** : −0,018 [−0,042 ; +0,005] avec trois faux |
| EE9 atome manquant signalé (≥ 80 %) | **non soutenue** : 58 % |
| EE10 hors-classe signalé (≥ 80 %) | **non soutenue** : 7 % ; les tâches récupérées sont bien KNOWN (100 %), mais presque toutes les autres aussi |
| EE11 `flat_scene_k3` meilleure source | **non soutenue** : `k5` fait un peu mieux (+0,009), les atomes du pipeline gelé font jeu égal |

## 11. Réponses

**1. Le moteur sur classes survit-il aux primitives découvertes ?**
Non. Avec les primitives découvertes il n'apporte aucun gain d'accord sur le pipeline gelé et il est moins bon que
l'aval actuel sur les mêmes primitives (−0,012). Il ne produit une classe correcte à 0,99 que dans 7 % des tâches,
toutes aux profondeurs 1 et 2. Sa force venait de l'exactitude des atomes : un bruit d'écart-type 0,10 sur les
paramètres des vrais atomes fait tomber la récupération de 0,79 à 0,02.

**2. flat_scene_k3 reste-t-il la meilleure source de primitives ?**
Il reste parmi les meilleures sans être seul en tête. Cinq hypothèses par type font légèrement mieux (+0,009), les
atomes du pipeline gelé font jeu égal, les trois autres sources sont nettement moins bonnes. Aucune source sans
oracle ne dépasse 0,50 de vrais atomes présents ; avec les vrais atomes le même moteur atteint 0,936.

**3. La réduction des littéraux conserve-t-elle les bons atomes ?**
Imparfaitement. La réduction retenue garde 71 % des vrais atomes présents, et seulement 87 % quand on ne lui donne
que des vrais. Elle rate les atomes qui n'agissent qu'en combinaison. Mais elle n'est pas le verrou : garder les
quatorze hypothèses avec le moteur large ne gagne que 0,007 d'accord.

**4. La profondeur 4–6 reste-t-elle viable ?**
Non, pour aucune des trois chaînes. Aucune tâche de profondeur 3 ou plus n'est récupérée à 0,99, et la structure
candidate à 0,97 n'est atteinte que dans 2 à 10 % des tâches de profondeur 4 à 6, selon la chaîne. La cause est mesurée : 23 % des
vrais atomes y sont présents dans les primitives.

**5. Les faux atomes sont-ils le nouveau goulot ?**
Non. Ajouter trois faux atomes aux vrais ne coûte que 2 points de récupération : le moteur les écarte. Ils sont
nombreux (cinq sur six à la profondeur 6), mais c'est le symptôme de l'absence des vrais, pas une cause propre.

**6. Les atomes manquants sont-ils le nouveau goulot ?**
Oui. « Primitive correcte absente » est la cause principale de 292 échecs sur 349, et de tous les échecs à partir de
la profondeur 4. En contrôlé, retirer un seul vrai atome fait tomber la récupération de 0,79 à 0,01. S'y ajoute la
précision : un atome présent mais approximatif ne suffit pas pour 0,99.

**7. Le score final est-il encore mal aligné ?**
Peu, et ce n'est plus ce qui limite. Erreurs et pénalité légère font jeu égal ; le score existant perd 0,012
d'accord. Aucun échec n'est attribué à un mauvais classement : quand la bonne classe existe, elle est choisie.

**8. L'ajustement reste-t-il destructeur ?**
Non, avec des atomes découverts il aide : +0,011 d'accord en version validée, et la version forcée fait jeu égal.
Trois échecs seulement lui sont attribués, à la profondeur 1. Le résultat du sous-jalon précédent (l'ajustement
détruit) vaut pour des atomes déjà exacts ; pour des atomes approximatifs, ajuster corrige plus qu'il n'abîme. Les
deux constats sont cohérents : l'ajustement rapproche de l'à-peu-près et éloigne de l'exact.

**9. Le système sait-il répondre UNKNOWN / NONE_OF_THE_ABOVE ?**
Non. Il répond KNOWN dans 93 % des tâches hors classe et dans 97 % des tâches en classe qu'il n'a pas récupérées.
L'infrastructure de frontière de classe existante n'a pas été réutilisée (elle demande un modèle ajusté sur
validation, qui n'existe pas) ; la décision à deux signaux évaluée ici ne suffit pas.

**10. La chaîne est-elle prête pour campagne confirmatoire ?**
Non. La chaîne C ne bat aucune des deux baselines. Je ne propose pas la campagne 16 graines.

## 12. Nouveau goulot

La découverte des atomes, en rappel et en précision. À la profondeur 6, une primitive découverte sur six correspond à
un vrai atome ; aux profondeurs 4 à 6, 23 %. Tout le reste de la chaîne a été montré fonctionnel à atomes exacts
(composition, 0,99) ou secondaire (réduction, score, ajustement, faux atomes). Les sous-jalons amont précédents
avaient mesuré ce même verrou par morceaux (paramètres sous masquage, attribution, type) sans le lever.

Deux résultats restent acquis :

- `flat_scene_k3` + aval actuel bat le pipeline gelé sur un troisième jeu indépendant (accord +0,019
  [+0,013 ; +0,026], structure candidate +0,083 [+0,050 ; +0,117]) ;
- la composition n'est pas le verrou : le moteur sur classes passe à quatorze atomes en un quart de seconde.

## 13. Limites

- Dev uniquement, une graine par tâche.
- Profondeur maximale 6 pour A et B, sans information sur la taille vraie : leurs chiffres ne sont pas directement
  comparables à ceux des sous-jalons précédents, qui donnaient cette information.
- Pour C, la « structure candidate » est restreinte aux 20 premières classes ; sur toute la table, une classe à 0,97
  n'existe de toute façon que dans 22 % des tâches.
- La configuration de C a été réglée sur 60 tâches ; d'autres réglages (plus d'hypothèses par type, moteur large) font
  un peu mieux en secondaire, sans approcher B.
- Les seuils de décision KNOWN / UNKNOWN / NONE_OF_THE_ABOVE n'ont pas été réglés.
- Le défaut de la réduction sur les atomes qui n'agissent qu'en combinaison a été constaté sur l'évaluation ; il n'a
  pas été corrigé, pour ne pas régler sur ces tâches.

## 14. Verdict

Règle pré-enregistrée : EE1 non soutenue.

`END-TO-END STILL BLOCKING`

La campagne confirmatoire 16 graines n'est pas proposée. Le lot scellé n'a pas été touché. Le pipeline gelé reste la
référence officielle, `flat_scene_k3` reste une baseline candidate, et rien de ce sous-jalon n'est branché.

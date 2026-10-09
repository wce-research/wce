# WCE Deep Composition & Final Reconstruction with Oracle Atoms — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucun test confirmatoire, aucune campagne 16 graines,
lot scellé jamais touché. Le pipeline gelé reste la référence officielle ; `flat_scene_k3` reste une baseline
candidate, tenue à l'écart de ce sous-jalon ; rien n'est branché par défaut.

Dans tout ce sous-jalon les **vrais atomes** (types et paramètres) sont fournis. L'amont est retiré de l'équation.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur la clé de composition 899 (six tâches, une par profondeur). Évaluation sur
**384 tâches jamais utilisées** : clés 800–831, profondeurs 1 à 6 (nombre d'atomes), régimes standard et rich.

Méthode principale : `dp_errors` — programmation dynamique ascendante sur les classes d'équivalence observationnelle
(masque de cellules), exhaustive jusqu'à la taille de la vérité, classe retenue = moins d'erreurs, puis plus petite ;
aucun ajustement de paramètres. Référence : `ref_b150` — pipeline existant, mêmes atomes, budget de référence.

| # | Hypothèse | Critère |
|---|---|---|
| DC1 | avec les vrais atomes, la classe de la vraie structure est toujours construite | part des tâches = 1,00, et moins de 10⁶ paires évaluées à la profondeur 6 |
| DC2 | la composition sur classes récupère mieux que le pipeline existant | récupération ≥ 0,99, `dp_errors` − `ref_b150`, IC 95 % > 0, globalement et à chaque profondeur 2–6 |
| DC3 | le classement n'est pas le verrou quand on classe par erreurs ; il l'est avec les pénalités actuelles | vraie classe au rang 1 par erreurs dans ≥ 90 % des tâches ; `dp_posterior` − `dp_errors` sur la récupération, IC < 0 |
| DC4 | l'ambiguïté intrinsèque est petite | masse des cellules jamais observées < 5 % en moyenne ; quand la classe retenue est celle de la vérité, accord ≥ 0,99 dans ≥ 95 % des cas |
| DC5 | le budget n'est pas le problème du pipeline existant | `ref_b450` − `ref_b150` et `ref_b1500` − `ref_b150` sur l'accord : IC incluant 0 ou effet < 0,01 |
| DC6 | l'ajustement du composite nuit | accord, C − A : IC < 0, sur la vraie structure ; D (validé) − A : IC incluant 0 ou négatif |
| DC7 | la rencontre au milieu atteint le résultat exhaustif avec bien moins d'évaluations | `mitm3` égale l'exhaustif dans ≥ 95 % des tâches |
| DC8 | un faisceau de 100 classes suffit | vraie classe gardée dans ≥ 95 % des tâches |
| DC9 | la forme de l'arbre ne change rien pour la composition sur classes | récupération chaîne contre équilibré : écart < 0,05 |
| DC10 | l'espace effectif reste compact | classes cumulées à la profondeur 6 < 10⁵ en moyenne ; moins d'une seconde |

Attente déclarée d'après la mise au point (six tâches) : toutes probables, sauf DC7 incertaine (la recherche guidée
s'arrête parfois sur une structure quasi équivalente) et DC8 incertaine à la profondeur 6.

Deux limites connues avant lancement :

- `dp_errors` reçoit le nombre d'atomes vrais comme taille maximale. La variante « une feuille de plus »
  (`dp_errors_plus1`, `dp_bic_plus1`) est rapportée à côté pour mesurer ce que vaut cette information.
- Avec des atomes exacts, « récupérer » signifie choisir la bonne fonction booléenne des atomes ; aucun paramètre
  n'est estimé. Ce résultat ne dit rien d'atomes imparfaits.

Règle de verdict fixée d'avance. Elle porte sur la composition à atomes donnés, et sur rien d'autre :

- `DEEP COMPOSITION READY FOR END-TO-END VALIDATION` : DC2 soutenue ET récupération ≥ 0,99 de `dp_errors` au moins
  égale à 0,80 à chaque profondeur de 2 à 6.
- `DEEP COMPOSITION IMPROVED BUT INSUFFICIENT` : DC2 soutenue, mais la barre de 0,80 manquée à au moins une profondeur.
- `DEEP COMPOSITION STILL BLOCKING` : DC2 non soutenue.

## 1. Protocole exécuté

- Run AWS `wce-deepdc-20261009-212700` (EC2 + SSM, 2 × c6i.xlarge, 28 et 23 min, 2 512 sous-unités, 0 erreur).
  Commandes : `make deep-run-dc`, `make deep-fetch`, `make deep-dc-aggregate`, `make deep-dc-report`.
  Résumé : `results/deep_composition_oracle_summary.json`.
- 384 tâches jamais utilisées : 64 par profondeur (nombre d'atomes) de 1 à 6, moitié standard (300 observations, 5 %
  de bruit), moitié rich (600 observations, 2 %). Vrais atomes fournis partout.
- Code : `core/deep_compose.py` (système : reçoit des atomes et des observations, ne voit ni vérité ni monde),
  `core/deep_compose_exp.py` (évaluation), `core/deep_dc_report.py`. Tests : 56 nouveaux (`make deep-dc-test`) ; suite
  complète 4 950 tests OK, 7 ignorés ; lot scellé intact.
- Aucun réglage sur l'évaluation. Une graine par tâche. Aucun guide appris n'a été introduit : les méthodes explicites
  n'ont pas plafonné.
- Écart au protocole : aucun. Défaut d'instrumentation relevé après coup : dans la loi d'échelle, la colonne des
  masques distincts répète celle des classes nouvelles ; elle n'est pas citée.

### Principe du moteur

Avec des atomes fixés, une composition booléenne est entièrement décrite par sa valeur sur les **cellules** : les
groupes d'observations qui ont le même vecteur de valeurs d'atomes. Six atomes donnent au plus 64 cellules, dont 21
sont occupées en moyenne à la profondeur 6. Une composition se résume à un masque de 64 bits. Ce masque sert à la fois
de signature fonctionnelle, de clé de classe d'équivalence observationnelle et de moyen de compter les erreurs sans
exécuter le programme.

## 2. Résultat principal

Récupération fonctionnelle (accord ≥ 0,99 sur toute la région) :

| Profondeur | Pipeline existant, vrais atomes (`ref_b150`) | Le même sans ajustement | Le même avec garde-fou | Composition sur classes (`dp_errors`) |
|---|---|---|---|---|
| 1 | 0,67 | 0,75 | 0,73 | 1,00 |
| 2 | 0,23 | 0,97 | 0,80 | 1,00 |
| 3 | 0,06 | 0,94 | 0,58 | 1,00 |
| 4 | 0,00 | 0,84 | 0,48 | 1,00 |
| 5 | 0,00 | 0,78 | 0,47 | 0,97 |
| 6 | 0,00 | 0,72 | 0,38 | 0,95 |
| toutes | 0,16 | 0,83 | 0,57 | **0,99** |

`dp_errors` − `ref_b150`, différence appariée par tâche, IC 95 % : récupération +0,826 [+0,787 ; +0,862] ; accord
+0,049 [+0,043 ; +0,055] ; accord hors de la région observée +0,057 [+0,046 ; +0,069]. L'intervalle est au-dessus de
zéro à chaque profondeur, de +0,33 [+0,22 ; +0,45] à la profondeur 1 à +0,95 [+0,89 ; +1,00] à la profondeur 6.

Accord moyen : `dp_errors` 0,9996 (0,9994 dans la région observée, 1,0000 hors région) ; `ref_b150` 0,951
(0,954 / 0,943).

## 3. Décomposition par profondeur

Composition sur classes, exhaustive jusqu'à la taille de la vérité :

| Profondeur | Vraie classe construite | Rang moyen par erreurs | Rang 1 | Arbre exact | Même fonction sur les observations | Fonctionnel | Accord |
|---|---|---|---|---|---|---|---|
| 1 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,0000 |
| 2 | 1,00 | 1,00 | 1,00 | 0,95 | 1,00 | 1,00 | 0,9999 |
| 3 | 1,00 | 1,02 | 0,98 | 0,73 | 0,98 | 1,00 | 0,9999 |
| 4 | 1,00 | 1,09 | 0,94 | 0,64 | 0,94 | 1,00 | 0,9996 |
| 5 | 1,00 | 1,06 | 0,94 | 0,39 | 0,94 | 0,97 | 0,9994 |
| 6 | 1,00 | 1,20 | 0,89 | 0,44 | 0,89 | 0,95 | 0,9985 |

Pipeline existant, mêmes atomes, trace par étape (160 tâches, profondeurs 2 à 6) :

| Profondeur | Bon candidat (≥ 0,99) avant ajustement | Son rang médian | Accord du premier candidat avant ajustement | Accord final | Final ≥ 0,99 | Cause dominante de la perte |
|---|---|---|---|---|---|---|
| 2 | 1,00 | premier | 1,000 | 0,979 | 0,31 | ajustement initial (16 sur 22 pertes) |
| 3 | 1,00 | premier | 0,999 | 0,970 | 0,03 | ajustement initial (25 sur 31) |
| 4 | 1,00 | premier | 0,991 | 0,936 | 0,00 | ajustement initial (30 sur 32) |
| 5 | 1,00 | premier | 0,991 | 0,932 | 0,00 | ajustement initial (32 sur 32) |
| 6 | 0,97 | premier | 0,979 | 0,907 | 0,00 | ajustement initial (31 sur 32) |

Sur les 160 tâches tracées : ajustement initial 134, récupérées 11, seuil trop strict 10, classement 3, autres 2.
La simplification ne change rien (accord identique avec et sans).

Pipeline existant, vivier final (384 tâches) :

| Bras | Vraie forme dans le vivier | Bon candidat dans le vivier | Bon candidat trouvé puis non retenu | Arbre exact retenu | Récupération |
|---|---|---|---|---|---|
| `ref_b150` | 0,84 | 0,20 | 0,04 | 0,49 | 0,16 |
| sans ajustement | 0,83 | 0,98 | 0,15 | 0,58 | 0,83 |
| avec garde-fou | 0,83 | 0,84 | 0,27 | 0,54 | 0,57 |

## 4. Espace de composition et loi d'échelle

| Atomes | Cellules occupées | Compositions évaluées | Classes d'équivalence | Temps (s) | Classes quasi équivalentes à la vérité | Classes strictement meilleures que la vérité |
|---|---|---|---|---|---|---|
| 1 | 2,0 | 1 | 1 | 0,0000 | 1,0 | 0,00 |
| 2 | 3,9 | 14 | 6 | 0,0002 | 1,1 | 0,00 |
| 3 | 6,4 | 106 | 28 | 0,0004 | 1,6 | 0,00 |
| 4 | 10,3 | 1 748 | 276 | 0,0009 | 2,6 | 0,08 |
| 5 | 15,5 | 26 173 | 4 007 | 0,005 | 4,4 | 0,03 |
| 6 | 20,7 | 671 588 | 71 257 | 0,11 | 6,5 | 0,17 |

Chaque atome supplémentaire multiplie le nombre de classes par un facteur qui croît : 5, 10, 14 puis 18. À la profondeur 6 : 71 000 classes en moyenne,
337 000 au maximum, 0,6 s au pire. « Compositions évaluées » compte les paires de classes combinées par un opérateur,
pas des programmes bruts : chaque sous-structure n'est construite qu'une fois.

Classes nouvelles par taille, avec six atomes : 6 ; 38 ; 250 ; 1 698 ; 10 510 ; 58 754.

Énumération exhaustive aux profondeurs 2 et 3 : 14 et 106 compositions, 6 et 28 classes ; la vraie classe est première
par erreurs dans 100 % et 98 % des tâches ; 1,1 et 1,6 classe est à moins de 3 % de la vérité sur les observations.

Signature sur les observations contre signature étendue à toute la région : à la profondeur 5, 4 007 classes contre
4 867, donc 18 % des classes distinctes ailleurs sont confondues sur les observations. L'effet fonctionnel est
négligeable : 0,2 % des points de la région tombent dans une cellule jamais observée (0,6 % à la profondeur 6).

## 5. Classement

Même table de classes, cinq sélecteurs (récupération fonctionnelle) :

| Sélecteur | Toutes | d4 | d5 | d6 | Rang moyen de la vraie classe |
|---|---|---|---|---|---|
| erreurs, puis taille | 0,987 | 1,00 | 0,97 | 0,95 | 1,06 |
| vraisemblance profilée | 0,987 | 1,00 | 0,97 | 0,95 | 1,06 |
| vraisemblance − ½·ln N par feuille | 0,984 | 0,98 | 0,97 | 0,95 | 1,06 |
| score a posteriori existant | 0,906 | 0,91 | 0,80 | 0,77 | 1,35 |
| plus petite structure | 0,167 | 0,00 | 0,00 | 0,00 | 2 130 |

Score a posteriori − erreurs : −0,081 [−0,109 ; −0,055] de récupération. Ses pénalités (structure et paramètres,
environ 13 nats par atome) font retenir une structure trop petite : taille moyenne 5,4 au lieu de 6 à la profondeur 6.

Sans connaître la taille vraie (une feuille de plus autorisée, profondeurs 1 à 5) : 0,991 par erreurs, 0,991 avec la
pénalité légère. L'information de taille ne compte pas.

## 6. Équivalence

| | Part des 384 tâches |
|---|---|
| arbre exact (à associativité et commutativité près) | 0,69 |
| arbre différent, même fonction sur les observations | 0,27 |
| fonction différente sur les observations, accord ≥ 0,99 | 0,03 |
| échec fonctionnel | 0,013 (5 tâches) |
| plus compact que la vérité et fonctionnel | 0,010 (4 tâches) |

Quand la classe retenue est celle de la vérité (368 tâches), l'accord est toujours ≥ 0,9957. Parmi les 118 tâches où
l'arbre n'est pas exact, 96 % sont fonctionnellement correctes. Les 16 tâches où la classe diffère sont celles où le
bruit rend une autre fonction aussi bonne ou meilleure sur les observations (11 fois strictement moins d'erreurs que
la vérité) ; 11 restent fonctionnelles.

Les 5 échecs : profondeurs 5 et 6, quatre chaînes et un arbre mixte, accords de 0,965 à 0,989. Dans l'un, 7,6 % de la
région est dans des cellules jamais observées.

## 7. Recherches bornées

Faisceau sur les classes (vraie classe gardée) :

| Faisceau | Toutes | d4 | d5 | d6 | Compositions évaluées à d6 |
|---|---|---|---|---|---|
| 10 classes | 0,72 | 0,70 | 0,45 | 0,17 | 2 034 |
| 100 classes | 0,95 | 1,00 | 0,88 | 0,80 | 62 289 |
| 1 000 classes | 0,98 | 1,00 | 0,97 | 0,89 | 387 053 |

Budget en compositions évaluées (vraie classe atteinte) :

| Budget | d5 | d6 | Accord moyen à d6 |
|---|---|---|---|
| 100 | 0,00 | 0,00 | 0,791 |
| 1 000 | 0,03 | 0,00 | 0,854 |
| 10 000 | 0,72 | 0,03 | 0,924 |
| 100 000 | 1,00 | 0,52 | 0,977 |
| exhaustif | 1,00 | 1,00 | 0,9985 |

Rencontre au milieu avec trous contraints :

| Table | Fonctionnel, toutes | d4 | d5 | d6 | Égale l'exhaustif | Jointures à d6 |
|---|---|---|---|---|---|---|
| taille ≤ 2 | 0,92 | 0,97 | 0,83 | 0,72 | 0,91 | 2 051 |
| taille ≤ 3 | 0,91 | 1,00 | 0,78 | 0,66 | 0,91 | 62 |

Contre l'exhaustif : −0,068 [−0,094 ; −0,042] et −0,081 [−0,109 ; −0,055] de récupération. L'accord moyen reste à
0,997 : la recherche guidée s'arrête sur des structures quasi équivalentes.

Forme de l'arbre (`dp_errors` / pipeline existant) : équilibré 1,00 / 0,00 ; chaîne 0,97 / 0,00 ; mixte 0,98 / 0,00.
Par nombre de OU exclusifs : 1,00 jusqu'à 1 ; 0,99 à 2 ; 0,94 à 3. Un seul opérateur : 1,00 ; opérateurs mêlés : 0,98.

## 8. Ajustement du composite

Départ : la VRAIE structure avec les VRAIS paramètres (accord 1,000).

| Variante | Accord moyen | Récupération ≥ 0,99 | Différence d'accord avec A, IC 95 % |
|---|---|---|---|
| A — aucun ajustement | 1,000 | 1,00 | — |
| B — chaque atome ré-ajusté seul | 0,867 | 0,08 | −0,133 [−0,143 ; −0,123] |
| C — ajustement du composite (actuel) | 0,962 | 0,17 | −0,038 [−0,042 ; −0,035] |
| D — le même, validé | 0,988 | 0,77 | −0,012 [−0,014 ; −0,009] |

Mêmes chiffres sur la structure retenue par `dp_errors`. L'ajustement C augmente même les erreurs sur les
observations (37,5 → 39,3 en moyenne). La version validée accepte 25 % des propositions ; ces acceptations font
baisser l'accord réel alors que le critère observable monte.

Le budget du pipeline existant n'est pas un facteur : les résultats à 50, 150, 450 et 1 500 sont identiques tâche par
tâche. Il s'arrête de lui-même vers 28 équivalents-ajustement, après avoir poli une vingtaine de candidats.

## 9. Hypothèses pré-enregistrées

| # | Résultat |
|---|---|
| DC1 vraie classe toujours construite, moins de 10⁶ paires à d6 | **soutenue** pour la construction (1,00) ; la borne tient en moyenne (0,67 million) et dans 77 % des tâches de profondeur 6, pas au pire (2,0 millions) |
| DC2 composition sur classes > pipeline existant | **soutenue** : +0,826 [+0,787 ; +0,862], et à chaque profondeur |
| DC3 classement | **soutenue** : rang 1 par erreurs dans 96 % des tâches ; score existant −0,081 [−0,109 ; −0,055] |
| DC4 ambiguïté intrinsèque petite | **soutenue** : 0,2 % de masse non observée ; accord ≥ 0,99 dans 100 % des cas où la classe est la bonne |
| DC5 le budget n'est pas le problème | **soutenue** : différence exactement nulle à 3 × et à 10 × |
| DC6 l'ajustement du composite nuit | **soutenue** : C − A −0,038 [−0,042 ; −0,035] ; D − A −0,012 [−0,014 ; −0,009] |
| DC7 rencontre au milieu = exhaustif dans ≥ 95 % | **non soutenue** : 0,91 (0,63 à la profondeur 6) |
| DC8 faisceau de 100 classes ≥ 95 % | **non soutenue** : 0,945 (0,80 à la profondeur 6) |
| DC9 forme de l'arbre indifférente | **soutenue** : équilibré 1,00, chaîne 0,97 |
| DC10 espace compact | **soutenue** : 71 000 classes et 0,11 s en moyenne à la profondeur 6 |

## 10. Réponses

**1. Pourquoi les vrais atomes ne suffisent-ils pas ?**
Parce que le pipeline les modifie. Avec les vrais atomes, un candidat à 0,99 ou plus existe avant ajustement dans 99 %
des tâches, et il est en tête du classement. L'ajustement initial le dégrade : c'est la cause de 134 pertes sur 149.
Même un atome seul, donné avec ses vrais paramètres, n'est rendu correct que deux fois sur trois. Le reste vient du
score, qui pénalise trop la taille. Ce n'est ni la composition ni l'espace de recherche.

**2. À quelle profondeur la composition s'effondre-t-elle ?**
Elle ne s'effondre pas quand rien n'est ajusté : 1,00 jusqu'à la profondeur 4, 0,97 à 5, 0,95 à 6. C'est le pipeline
existant qui s'effondre, dès la profondeur 2 (0,23) et totalement à partir de 4 (0,00).

**3. La vraie structure est-elle générée ?**
Oui. La classe de la vraie structure est construite dans 100 % des tâches par la composition sur classes. Dans le
pipeline existant, la vraie forme est dans le vivier dans 84 % des tâches (47 % à la profondeur 6).

**4. Est-elle perdue au ranking ?**
Pas avec un classement par erreurs : rang 1 dans 96 % des tâches, rang moyen 1,06. Oui, en partie, avec le score a
posteriori existant : il perd 8 points de récupération, 18 à la profondeur 6, en préférant des structures trop petites.

**5. L'équivalence fonctionnelle explique-t-elle une partie de l'échec ?**
Elle explique les écarts de structure, pas des échecs. Dans 27 % des tâches l'arbre retenu diffère de la vérité tout
en calculant la même fonction, et dans 3 % une fonction légèrement différente reste correcte. Compter « arbre exact »
ferait croire à 31 % d'échecs là où il y en a 1,3 %. L'ambiguïté réellement intrinsèque (cellules jamais observées)
pèse 0,2 % de la région.

**6. Le dynamic programming aide-t-il ?**
Oui, c'est ce qui rend le problème petit. Construire chaque sous-structure une fois et ne garder qu'une structure par
classe ramène la profondeur 6 à 71 000 classes et un dixième de seconde.

**7. Le meet-in-the-middle aide-t-il ?**
Non, pas ici. Il utilise 10 à 10 000 fois moins d'évaluations, mais il perd 7 à 8 points de récupération et n'égale
l'exhaustif que dans 63 à 67 % des tâches à la profondeur 6. L'exhaustif coûtant déjà un dixième de seconde, il n'y a
rien à gagner jusqu'à six atomes.

**8. Le problème est-il surtout budget ou stratégie ?**
Ni l'un ni l'autre pour le pipeline existant : dix fois plus de budget donne exactement le même résultat, parce qu'il
ne consomme pas son budget. Pour la composition sur classes, le budget compte à la profondeur 6 : 100 000
compositions n'atteignent la vraie classe qu'une fois sur deux, il en faut 670 000 en moyenne. C'est un coût, pas un
obstacle.

**9. Le fit composite aide-t-il ou nuit-il ?**
Il nuit, sans exception mesurée. Partant de la vraie structure et des vrais paramètres, il fait tomber la
récupération de 1,00 à 0,17. Validé, il remonte à 0,77 mais reste sous l'absence d'ajustement (−0,012
[−0,014 ; −0,009]) : les propositions qu'il accepte améliorent le critère observable et dégradent l'accord réel.

**10. Peut-on rendre la composition profonde viable sans brute force ?**
Oui, à atomes donnés et jusqu'à six atomes. L'énumération porte sur les classes d'équivalence, pas sur les programmes,
et reste exhaustive pour un coût négligeable. La limite est structurelle : le masque suppose au plus six atomes, et le
nombre de classes est multiplié par un facteur croissant (18 du cinquième au sixième atome). Au-delà, il faudra un faisceau, qui perd déjà 20 % des
vraies classes à la profondeur 6 avec 100 classes gardées.

**Réponse à la question du sous-jalon.** La profondeur n'est difficile ni parce que l'espace explose, ni parce que
des structures sont intrinsèquement équivalentes. Avec les vrais atomes, elle était difficile parce que l'ajustement
détruisait les bonnes solutions et que le score préférait des structures trop petites.

### Ce que cela implique pour `flat_scene_k3` (non mesuré)

Rien de ce sous-jalon n'a été exécuté sur `flat_scene_k3`. Deux implications à vérifier, pas des résultats :

- `flat_scene_k3` passe par le même ajustement et le même score que le pipeline existant. Si la dégradation mesurée
  ici s'y retrouve, une partie de son écart aux vrais atomes viendrait de l'aval et non de ses primitives.
- Le moteur sur classes ne lui est pas applicable tel quel : il fournit une vingtaine de littéraux, le masque en
  accepte six. Et avec des littéraux imparfaits, « ne pas ajuster » n'est plus évidemment le bon choix.

## 11. Limites

- Atomes exacts : aucun paramètre n'est estimé. Le résultat ne dit pas ce que devient la composition sur classes avec
  des atomes approximatifs, ni avec des atomes faux mêlés aux vrais.
- Au plus six atomes, par construction du masque.
- La récupération par erreurs suppose que tous les atomes fournis sont utiles ; avec des atomes superflus, minimiser
  les erreurs sur-ajusterait. La pénalité légère par feuille fait aussi bien ici (0,984), mais n'a pas été mise à
  l'épreuve d'atomes superflus.
- Dev uniquement, une graine par tâche, pas de test confirmatoire.
- Le pipeline existant n'a été comparé que dans sa stratégie factorisée, la seule qui accepte des atomes fournis.
- La trace par étape porte sur 160 des 384 tâches (clés paires, profondeurs 2 à 6).
- Rencontre au milieu et faisceau : mises en œuvre simples ; leur échec relatif vaut pour ces versions.

## 12. Verdict

Règle pré-enregistrée : DC2 soutenue, et récupération de `dp_errors` de 1,00 ; 1,00 ; 1,00 ; 0,97 ; 0,95 aux profondeurs
2 à 6, donc au-dessus de 0,80 partout.

`DEEP COMPOSITION READY FOR END-TO-END VALIDATION`

Portée exacte de ce verdict : la composition, **à atomes exacts fournis**, n'est plus un verrou et peut entrer dans
une validation de bout en bout. Il ne dit pas que la chaîne complète est prête : l'amont reste au niveau du sous-jalon
précédent, et l'ajustement des paramètres, tel qu'il existe, détruit les solutions. Le pipeline gelé reste la référence
officielle ; rien de ce sous-jalon n'y est branché.

Portée enregistrée de ce résultat : vrais atomes ; vrais paramètres ; jusqu'à 6 atomes ; aucun résultat confirmatoire
global ; aucun lot scellé. La composition sur classes n'est pas le pipeline officiel.

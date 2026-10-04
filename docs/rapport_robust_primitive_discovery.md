# WCE Robust Primitive Discovery — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucun test confirmatoire, aucune campagne 16 graines,
lot scellé jamais touché. Le pipeline gelé reste la référence officielle ; la porte conditionnelle n'est pas active par
défaut. Seul l'amont est modifié :

```
observations → primitive proposal → primitive validation → primitive parameter estimation → primitive set
```

Composition, score final, porte, simplification et sélection sont ceux du pipeline existant, inchangés.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur les scènes d'indices 5000+ (27 à 36 scènes) et la tâche de composition 499.
Évaluation sur des scènes jamais vues (indices 2000–2031, 9 catégories × 32 = 288 scènes) et sur 128 tâches de
composition jamais utilisées (clés 400–415).

Bras principal : `union_joint` (proposeurs consensus local + vote de Hough, estimation locale, validateur calibré,
élimination arrière, assemblage conjoint). Référence : `gradient` (gradient direct, validateur précédent).

| # | Hypothèse | Critère |
|---|---|---|
| RP1 | le rappel d'équivalence dépasse celui de la référence | différence appariée par scène, IC 95 % au-dessus de 0 |
| RP2 | les faux positifs par scène baissent d'au moins la moitié | différence appariée, IC 95 % sous 0, et ratio ≤ 0,5 |
| RP3 | anneaux et secteurs : rappel d'équivalence supérieur à la référence | par type d'atome |
| RP4 | l'estimation locale des paramètres améliore le rappel à proposeur égal | `ransac_local_rv` − `ransac_rv` |
| RP5 | le validateur calibré a un taux de fausse acceptation inférieur à l'ancien | étude sur vivier commun |
| RP6 | l'assemblage conjoint retrouve des atomes riches exacts sans dégrader la précision | `union_joint` − `union_rv` |
| RP7 | les mesures actives dirigées battent les mesures aléatoires | équivalence, même budget |
| RP8 | gain aval : structure candidate et récupération finale supérieures à la référence | différences appariées par tâche, IC 95 % au-dessus de 0 |

Attente déclarée d'après la mise au point (27 scènes) : RP2 probable ; RP1 incertaine (rappel voisin de la référence) ;
RP3 incertaine ; RP8 inconnue.

Règle de verdict fixée d'avance :

- `PRIMITIVE DISCOVERY READY FOR END-TO-END VALIDATION` : RP1, RP2 et RP8 soutenues.
- `PRIMITIVE DISCOVERY IMPROVED BUT INSUFFICIENT` : RP1 ou RP2 soutenue, mais pas RP8, ou rappel d'équivalence ≤ 0,56.
- `PRIMITIVE DISCOVERY STILL BLOCKING` : ni RP1 ni RP2.

## 1. Protocole exécuté

- Run AWS `wce-deeprp-20261004-194145` (EC2 + SSM, 2 × c6i.xlarge, 49 min chacun, 2 608 sous-unités, 0 erreur,
  13 714 lignes). Commandes : `make deep-run-rp DEEP_RP_KEYS=0-15,400-415`, `make deep-fetch`, `make deep-rp-aggregate`,
  `make deep-rp-report`. Résumé : `results/deep_robust_primitive_summary.json`.
- Banc atomique : 288 scènes jamais vues (9 catégories × 32), 300 observations chacune. Chaîne aval : 128 tâches de
  composition jamais utilisées (profondeurs 2 à 5, régimes standard et rich).
- Réglage : `make deep-rp-tune` sur 27 scènes de mise au point → `results/deep_configs/robust_validator.json`
  (toutes les données, seuil de contraste 10,88 calibré sur étiquettes aléatoires, support ≥ 5, pureté ≥ 0,55,
  2 tronçons, élimination arrière à 3).
- Tests : 79 nouveaux (`make deep-rp-test`), suite complète 4 710 tests OK, 7 ignorés ; lot scellé intact.
- Écart au protocole : aucun. Limite : une seule graine par scène, pas de test confirmatoire.

## 2. Résultats principaux (banc atomique, 288 scènes)

| Bras | Exact | Équivalence | Précision | Faux positifs / scène | Retenues | Équiv.-ajust. | Secondes |
|---|---|---|---|---|---|---|---|
| `gradient` (référence) | 0,33 | **0,61** | 0,39 | 1,39 | 2,13 | 38,8 | 1,07 |
| `gradient_min` (base minimale, ancien validateur) | 0,33 | 0,34 | 0,52 | 0,90 | 1,62 | 13,7 | 0,33 |
| `gradient_rv` (gradient + validateur calibré) | 0,27 | 0,45 | 0,78 | 0,16 | 0,68 | 26,8 | 0,79 |
| `ransac_rv` | 0,30 | 0,53 | 0,73 | 0,30 | 1,27 | 0 | 0,14 |
| `ransac_local_rv` | 0,30 | 0,54 | 0,72 | 0,33 | 1,33 | 0 | 0,22 |
| `hough_rv` | 0,25 | 0,46 | 0,77 | 0,25 | 1,13 | 8,6 | 0,25 |
| `grid_rv` | 0,24 | 0,45 | 0,74 | 0,26 | 1,08 | 12,2 | 0,27 |
| `neural_rv` | 0,03 | 0,06 | 0,97 | 0,00 | 0,10 | 0 | 0,04 |
| `union_old` (nouveaux proposeurs, ancien validateur) | 0,34 | 0,58 | 0,30 | 3,05 | 4,38 | 19,5 | 0,65 |
| `union_rv` | 0,32 | 0,57 | 0,74 | 0,33 | 1,40 | 8,8 | 0,55 |
| **`union_joint`** (bras principal) | **0,44** | 0,57 | **0,74** | **0,35** | 1,74 | 9,0 | 0,57 |
| `union_all_joint` (+ grille, + gradient minimal) | 0,45 | 0,58 | 0,72 | 0,38 | 1,78 | 34,0 | 1,23 |
| `union_neural_joint` | 0,45 | 0,58 | 0,75 | 0,35 | 1,76 | 8,9 | 0,69 |
| `union_stab` (+ stabilité) | 0,44 | 0,57 | 0,74 | 0,35 | 1,75 | 9,0 | 10,58 |
| `union_holdout` (coupe par points) | 0,10 | 0,16 | 0,85 | 0,04 | 0,32 | 8,6 | 0,48 |
| `union_pairsplit` (coupe par paires) | 0,22 | 0,29 | 0,84 | 0,07 | 0,69 | 8,7 | 0,53 |

Différences appariées `union_joint` − `gradient`, IC 95 % bootstrap sur 288 scènes :

- équivalence : −0,04 [−0,09 ; +0,01] ;
- précision : +0,33 [+0,27 ; +0,38] ;
- faux positifs par scène : −1,03 [−1,17 ; −0,90] (ratio 0,25).

Rappel d'équivalence par type d'atome (référence → `union_joint`) : demi-plan 0,76 → 0,59 ; bande 0,52 → 0,42 ;
disque 0,59 → 0,66 ; anti-disque 0,45 → 0,47 ; anneau 0,43 → 0,46 ; secteur 0,12 → 0,00.

Par catégorie (équivalence, référence → `union_joint`) : single 0,97 → 0,84 ; two 0,33 → 0,41 ; multi 0,18 → 0,13 ;
close 0,47 → 0,63 ; nested 0,69 → 0,70 ; redundant 0,97 → 0,88 ; noise 0,59 → 0,22 ; outliers 0,72 → 0,72 ;
partial 0,59 → 0,63.

### Hypothèses pré-enregistrées

| # | Résultat |
|---|---|
| RP1 rappel d'équivalence supérieur | **non soutenue** : −0,04 [−0,09 ; +0,01] |
| RP2 faux positifs divisés par deux | **soutenue** : 1,39 → 0,35, IC sous 0 |
| RP3 anneaux et secteurs | **non soutenue** : anneaux 0,43 → 0,46, secteurs 0,12 → 0,00 |
| RP4 estimation locale | **non soutenue** : 0,53 → 0,54 à proposeur égal |
| RP5 fausses acceptations du validateur | **soutenue** : 16,9 % → 1,5 %, au prix des vraies acceptations (72,6 % → 31,6 %) |
| RP6 assemblage conjoint | **soutenue** : exact 0,32 → 0,44, précision inchangée (0,74) |
| RP7 mesures actives | **non soutenue** : dirigées 0,42, aléatoires 0,38, aucune mesure 0,43 (48 scènes, sans IC) |
| RP8 gain aval | **non soutenue** (voir question 9) |

## 3. Audit de la référence (646 composantes de frontière vraies)

| Classe | `gradient` | `union_joint` |
|---|---|---|
| trouvée | 54,8 % | 48,1 % |
| équivalente non reconnue par la métrique | 6,0 % | 2,8 % |
| vraie dominée par un autre candidat | 5,0 % | 5,7 % |
| vraie rejetée par le validateur | 9,9 % | **32,2 %** |
| paramètres éloignés | **18,0 %** | 9,9 % |
| mauvais type | 0,6 % | 0,9 % |
| jamais proposée | 5,7 % | 0,3 % |

Faux positifs de la référence : 1,39 par scène, dont 12 % proches d'une vraie frontière ; ce sont surtout des
secteurs (184 sur 400), des anneaux (63) et des bandes (63) — les atomes riches à paramètres libres nombreux.
Faux positifs de `union_joint` : 0,35 par scène, 46 % proches d'une vraie frontière, presque tous des disques.

Par type, pour la référence : les secteurs sont trouvés à 22 % (27 % paramètres éloignés, 21 % rejetés, 17 % non
reconnus) ; les anti-disques ne sont jamais proposés dans 18 % des cas.

## 4. Validateur : même vivier de 21 399 candidats

| Validateur | Vraies acceptations | Fausses acceptations | Précision des acceptations |
|---|---|---|---|
| ancien | 72,6 % | 16,9 % | 0,41 |
| toutes données, seuil nominal 3 | 94,2 % | 58,9 % | 0,20 |
| hors-échantillon (points), seuil nominal 3 | 71,9 % | 22,0 % | 0,34 |
| hors-échantillon, seuil calibré | 41,8 % | 4,7 % | 0,59 |
| toutes données, seuil calibré | 33,8 % | 1,7 % | 0,76 |
| configuration retenue | 31,6 % | 1,5 % | 0,77 |
| + stabilité | 31,4 % | 1,4 % | 0,78 (50 fois plus lent) |

Le contraste « z » n'est pas une normale réduite : les paires voisines partagent des points. Sur étiquettes aléatoires,
son maximum atteint 5,6 hors-échantillon et environ 11 sur toutes les données. Un seuil nominal de 3 accepte donc des
primitives sans aucune frontière. La coupe par paires n'est pas indépendante (un point bruité est dans les deux
moitiés) ; la coupe par points l'est, mais à 300 observations elle ne laisse plus assez de puissance (rappel 0,16).

## 5. Diversité des proposeurs (premier tour, 646 composantes)

| Proposeur | Rappel des propositions | Composantes uniques | Coût (équiv.-ajust.) |
|---|---|---|---|
| consensus local (`ransac`) | 0,77 | 0 | 0 |
| + estimation locale | 0,78 | 1 | 0 |
| gradient direct (riche) | 0,61 | 25 | 16,0 |
| Hough | 0,52 | 7 | 6,3 |
| géométrique (précédent) | 0,52 | 2 | 0 |
| grille | 0,48 | 5 | 6,4 |
| neuronal | 0,44 | 12 | 0 |
| gradient minimal | 0,33 | 0 | 6,0 |
| contraste | 0,20 | 0 | 7,2 |

Union de tous : 0,90. Redondants (aucune composante unique, retirables) : contraste, gradient minimal. Le consensus
local est le meilleur proposeur seul et ne coûte aucun ajustement.

## 6. Chaîne aval et plafonds oracle (128 tâches de composition)

| Littéraux | Équivalence des primitives | Précision | Structure candidate ≥ 0,97 | Final ≥ 0,95 | Final ≥ 0,99 | Accord moyen |
|---|---|---|---|---|---|---|
| vrais atomes | 1,00 | 1,00 | 0,66 | 0,73 | 0,05 | 0,954 |
| vrais atomes + garde-fou | 1,00 | 1,00 | 0,91 | 0,82 | 0,57 | 0,974 |
| vrais centres et orientations, le reste estimé | 0,37 | 0,43 | 0,45 | 0,46 | 0,03 | 0,908 |
| vrais types, paramètres estimés | 0,18 | 0,23 | 0,29 | 0,30 | 0,03 | 0,881 |
| pipeline gelé (référence) | 0,51 | 0,18 | 0,29 | 0,29 | 0,05 | 0,878 |
| gradient + ancien validateur | 0,19 | 0,20 | 0,22 | 0,27 | 0,02 | 0,854 |
| robuste, base minimale | 0,23 | 0,69 | 0,27 | 0,34 | 0,03 | 0,717 |
| robuste + assemblage conjoint | 0,23 | 0,69 | 0,25 | 0,34 | 0,04 | 0,717 |
| robuste + assemblage + tours résiduels | 0,23 | 0,69 | 0,30 | 0,37 | 0,04 | 0,723 |
| robuste + assemblage + garde-fou | 0,23 | 0,69 | 0,20 | 0,30 | 0,05 | 0,715 |

`robuste + assemblage` − référence, IC 95 % apparié : équivalence des primitives −0,28 [−0,33 ; −0,22] ; structure
candidate −0,04 [−0,11 ; +0,02] ; accord moyen −0,16 [−0,22 ; −0,11].

`POST-HOC / EXPLORATORY` : dans 26 tâches sur 128 la découverte robuste ne retient aucune primitive, et l'accord vaut
alors 0 par construction. Hors ces 26 tâches, l'accord moyen est 0,900 contre 0,900 pour la référence sur les mêmes
tâches, et la structure candidate 0,31 contre 0,33.

## 7. Réponses

**1. Pourquoi le gradient direct reste-t-il la meilleure baseline ?**
Son rappel (0,61) tient à deux choses mesurées. Il est le seul proposeur à apporter des composantes que les autres
n'atteignent pas (25 uniques), parce qu'il ajuste la cible et non la frontière. Et son validateur est permissif :
il accepte 73 % des vrais candidats, mais aussi 17 % des faux. Le prix est 1,39 faux positif par scène et une précision
de 0,39. Aucun bras ne dépasse son rappel ; `union_joint` l'égale à l'incertitude près (−0,04 [−0,09 ; +0,01]) avec
quatre fois moins de faux positifs et quatre fois moins d'équivalents-ajustement (temps mur divisé par deux).

**2. Pourquoi anneaux et secteurs échouent-ils ?**
Anneaux : le cercle intérieur est petit, donc peu de paires le portent ; il est proposé mais rejeté (22 %) ou estimé
trop loin (18 %). Le rappel reste à 0,46. L'assemblage conjoint fait passer la récupération exacte de l'anneau de 0,00
à 0,43 quand ses deux cercles sont trouvés.
Secteurs : un secteur est deux demi-droites, soit la moitié du contraste d'une droite. Avec le seuil calibré, 61 % de
leurs composantes sont rejetées et aucun secteur n'est retrouvé (0,00, contre 0,12 pour la référence). C'est une
régression nette du nouveau validateur sur ce type.

**3. La base minimale est-elle supérieure à la base riche ?**
Non, pas comme proposeur par gradient : 0,34 contre 0,61 en équivalence. Oui comme espace d'hypothèses pour les
proposeurs géométriques : le consensus local, qui ne propose que droites et cercles, atteint 0,77 de rappel de
proposition contre 0,61, sans ajustement. En aval, base minimale seule et base minimale + atomes assemblés donnent le
même résultat (accord 0,717). Le résultat du sous-jalon précédent (« minimale ≈ riche ») ne se reproduit donc pas pour
le gradient sur ces 288 scènes.

**4. Le validateur peut-il réduire fortement les faux positifs ?**
Oui : fausses acceptations de 16,9 % à 1,5 % sur le même vivier, faux positifs retenus de 1,39 à 0,35 par scène. Mais
il rejette aussi deux tiers des vrais candidats, et 32 % des composantes vraies sont désormais perdues par rejet
(contre 10 %). À 300 observations, on ne peut pas avoir à la fois un validateur calibré et le rappel de la référence.
La pureté, la cohérence multi-région et la stabilité n'ajoutent presque rien au seuil calibré.

**5. Les ensembles d'hypothèses aident-ils ?**
Non mesurable en gain. `CompatiblePrimitiveHypotheses` garde l'atome assemblé en concurrence avec ses parties, et les
littéraux transmis contiennent les deux ; en aval, cela ne change rien (0,717 dans les deux cas). L'objet existe et
est testé, son utilité n'est pas démontrée.

**6. Le joint fitting aide-t-il ?**
Partiellement. Il retrouve des atomes riches exacts (exact 0,32 → 0,44, bandes 0,00 → 0,42, anneaux 0,00 → 0,43)
sans coût en précision. Il ne change ni le rappel d'équivalence (0,57) ni le résultat aval. La règle de parenté
géométrique (seuil abaissé pour un cercle concentrique ou une droite parallèle) n'a pas été isolée par un bras dédié.

**7. Les mesures actives deviennent-elles utiles ?**
Non. Avec 24 mesures : dirigées 0,42, aléatoires 0,38, aucune 0,43. Ni l'une ni l'autre ne bat l'absence de mesure ;
48 scènes, pas d'intervalle. Elles ne sont pas intégrées.

**8. Un composant neuronal apporte-t-il enfin quelque chose ?**
Non. Seul, avec le validateur calibré, il retrouve 6 % des atomes. Ajouté à l'union, il donne +0,01 d'équivalence
(0,57 → 0,58), dans le bruit. Il apporte 12 composantes uniques en proposition, qui ne survivent pas à la validation.
Abandonné pour ce sous-jalon ; aucun réentraînement n'a été fait.

**9. Le gain primitive se traduit-il en gain structurel ?**
Non. Le gain obtenu est un gain de précision, pas de rappel, et il ne se transmet pas : structure candidate 0,25
contre 0,29, accord final moyen 0,72 contre 0,88. Sur les tâches de composition, la découverte robuste ne retrouve
que 23 % des primitives (les frontières y sont partiellement masquées par les ET / OU, donc sous le seuil calibré) et
ne retient rien dans 20 % des tâches. Le pipeline gelé, avec ses 84 littéraux peu précis, fait mieux : la composition
tolère les faux littéraux, pas les littéraux manquants.

**10. Quel est le nouveau goulot ?**
L'estimation des paramètres sous masquage, à nombre d'observations fixé. Les plafonds oracle le montrent : avec les
vrais atomes, la structure candidate est atteinte dans 66 % des tâches (91 % avec garde-fou) ; avec les vrais types
mais des paramètres à estimer, on retombe à 29 %, exactement le niveau de la référence ; donner en plus les centres et
orientations ne remonte qu'à 45 %. Connaître le type ne sert à rien si les paramètres ne sont pas trouvés. Second
goulot, déjà connu : même avec les vrais atomes, la récupération à 0,99 n'est que de 5 % sans garde-fou et 57 % avec.

## 8. Limites

- Dev uniquement, une graine par scène, aucune confirmation. Les seuils ont été réglés sur 27 scènes.
- Le rappel de la référence est 0,61 ici contre 0,56 au sous-jalon précédent : scènes différentes, pas une amélioration.
- La métrique d'équivalence a des tolérances fixes ; 3 à 6 % des composantes sont fonctionnellement retrouvées sans
  être reconnues.
- Les plafonds « vrais types » et « centres et orientations » estiment par contraste depuis des initialisations
  génériques ; un meilleur estimateur déplacerait ces plafonds.
- L'estimation locale conserve une vraie primitive dans 86 à 96 % des cas en mise au point, mais n'améliore pas le
  rappel de bout en bout (RP4) : le gain est absorbé par le rejet du validateur.
- Mesures actives et composant neuronal : évaluations légères, suffisantes pour ne pas les intégrer, pas pour conclure
  qu'ils sont inutiles en général.

## 9. Verdict

Règle pré-enregistrée : RP2 soutenue, RP1 et RP8 non soutenues.

`PRIMITIVE DISCOVERY IMPROVED BUT INSUFFICIENT`

L'amélioration est réelle mais étroite : mêmes primitives retrouvées, quatre fois moins de fausses, quatre fois moins
d'équivalents-ajustement, atomes riches exacts par assemblage. Elle ne dépasse pas le rappel de 0,56–0,61, régresse sur les secteurs
et les scènes bruitées, et dégrade l'aval. Le pipeline gelé reste la référence ; rien de ce sous-jalon n'y est branché.

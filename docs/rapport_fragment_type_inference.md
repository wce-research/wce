# WCE Fragment Attribution & Primitive Type Inference — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucun test confirmatoire, aucune campagne 16 graines,
lot scellé jamais touché. Le pipeline gelé reste la référence ; rien de ce sous-jalon n'y est branché par défaut.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur les scènes de la sous-plage dev + 350 000 (18 scènes par condition au plus) et sur
la tâche de composition 699. Évaluation sur 1 920 scènes de masquage jamais vues (indices 100–131 de la sous-plage
dev + 300 000) et sur 128 tâches de composition jamais utilisées (clés 600–615).

Méthode principale : `hier` (fragments par consensus pouvant se recouvrir, graphe de relations, familles puis types,
score de type longueur de description avec preuve négative limitée aux zones observables). En aval : `hier_k3`, sans
aucun oracle. Référence : pipeline gelé.

| # | Hypothèse | Critère |
|---|---|---|
| FT1 | l'attribution après inférence de type bat le regroupement par proximité | F1 par paires, différence appariée, IC 95 % > 0 |
| FT2 | la hiérarchie géométrie → famille → type bat la classification plate sur fragments | `hier` − `flat_fragments`, type correct, IC > 0 |
| FT3 | l'inférence sur fragments bat l'essai de tous les types sur la scène entière | `hier` − `flat_scene`, type correct, IC > 0 |
| FT4 | garder trois types par groupe retrouve nettement plus souvent le bon type | `type_3` − `type_1` ≥ 0,10 |
| FT5 | la preuve négative limitée aux zones observables aide | `hier` − `hier_no_negative` et `hier` − `hier_naive_negative`, IC > 0 |
| FT6 | la boucle attribution ↔ type aide | `hier_loop` − `hier`, IC > 0 |
| FT7 | la propriété du masquage reconnaît une frontière cachée | part cachée étiquetée « occluded » ≥ 0,5 à 40 % de masque |
| FT8 | les secteurs deviennent récupérables | récupération exacte ≥ 0,5 sans masque |
| FT9 | la mesure active dirigée lève l'ambiguïté de type mieux que l'aléatoire | différence appariée, IC > 0 |
| FT10 | aval sans oracle : structure candidate au-dessus du pipeline gelé | `hier_k3` − `reference`, IC > 0 |
| FT11 | aval sans oracle : accord final au-dessus du pipeline gelé | `hier_k3` − `reference` sur `final_full`, IC > 0 |
| FT12 | top-3 de types aide l'aval | `hier_k3` − `hier_k1` sur `final_full`, IC > 0 |

Attente déclarée d'après la mise au point : FT1, FT3, FT4 probables ; FT2, FT5, FT6 probablement nulles ; FT7 et FT8
probablement non atteintes ; FT10–FT12 inconnues (une tâche de mise au point, non concluante).

La matrice oracle (fragments vrais ou estimés × type vrai ou inféré) sert à répartir la perte ; elle n'est pas une
hypothèse.

Règle de verdict fixée d'avance. Cette fois, les deux bras comparés n'ont **aucun oracle** :

- `FRAGMENT/TYPE INFERENCE READY FOR END-TO-END VALIDATION` : FT10 et FT11 soutenues, avec un bras sans oracle.
- `FRAGMENT/TYPE INFERENCE IMPROVED BUT INSUFFICIENT` : FT1 ou FT3 soutenue, mais FT10 ou FT11 non soutenue.
- `FRAGMENT/TYPE INFERENCE STILL BLOCKING` : ni FT1 ni FT3.

Si un autre bras sans oracle que `hier_k3` (par exemple `flat_scene_k3` ou une variante avec tours résiduels) bat le
pipeline gelé alors que `hier_k3` ne le bat pas, ce sera rapporté comme résultat exploratoire, sans changer le verdict.

## 1. Protocole exécuté

- Run AWS `wce-deepft-20261005-110621` (EC2 + SSM, 2 × c6i.xlarge, 81 et 83 min, 2 976 sous-unités, 0 erreur, 33 920 lignes).
  Commandes : `make deep-run-ft`, `make deep-fetch`, `make deep-ft-aggregate`, `make deep-ft-report`.
  Résumé : `results/deep_fragment_type_summary.json`.
- Banc de masquage : 1 920 scènes jamais vues (6 types × 5 niveaux de masque × 2 niveaux de bruit × 32 primitives),
  une primitive cible et, sauf à 0 %, un masque. Chaîne aval : 128 tâches de composition jamais utilisées.
- Code : `core/deep_fragments.py` (système, sans vérité, sans modèle appris), `core/deep_fragments_exp.py`
  (évaluation), `core/deep_ft_report.py`. Tests : 57 nouveaux (`make deep-ft-test`) ; suite complète 4 840 tests OK,
  7 ignorés ; lot scellé intact.
- Aucun seuil réglé sur l'évaluation. Une seule graine par scène ; pas de test confirmatoire.
- Rappel de vocabulaire : dans la grammaire WCE un secteur est un angle non borné (sommet et deux demi-droites) ; il
  n'a pas d'arc. La branche secteur porte donc sur les demi-droites et le sommet, et sur la confusion avec un disque.

## 2. Attribution des fragments (1 920 scènes)

Paires de milieux observés : « même groupe » contre « même frontière vraie ».

| Regroupement | Précision | Rappel | F1 | Groupes | Groupes mêlant cible et masque |
|---|---|---|---|---|---|
| proximité | 0,83 | 0,51 | 0,59 | 5,1 | 31 % |
| + continuité tangentielle | 0,87 | 0,46 | 0,57 | 6,9 | 27 % |
| + centre compatible | 0,87 | 0,46 | 0,57 | 6,9 | 27 % |
| + orientation compatible | 0,87 | 0,48 | 0,59 | 6,8 | 27 % |
| + courbure compatible | 0,78 | 0,60 | 0,64 | 5,6 | 41 % |
| contraintes partagées (les trois) | 0,78 | 0,60 | 0,64 | 5,6 | 41 % |
| consensus (cercles et droites) | 0,65 | 0,82 | **0,71** | 2,9 | 60 % |
| après inférence de type | 0,62 | 0,85 | 0,70 | 1,3 | 65 % |

Différences appariées de F1 contre la proximité : consensus +0,11 [+0,10 ; +0,12] ; après inférence de type +0,11
[+0,10 ; +0,12] ; courbure +0,04 [+0,04 ; +0,05] ; continuité tangentielle −0,02 [−0,03 ; −0,02].

Deux régimes s'opposent. Les regroupements locaux sont purs mais morcellent chaque frontière en six ou sept
morceaux. Les regroupements par modèle rassemblent chaque frontière mais mêlent cible et masque dans 60 % des groupes.
Aucun mécanisme n'obtient à la fois la pureté et le rappel.

## 3. Inférence de type (1 920 scènes)

`type correct` : le type prédit pour le groupe de milieux attribué à la cible est le bon. Les colonnes top-1 et top-3
comptent parmi les n ou 3n premières hypothèses, n étant le nombre de primitives vraies de la scène.

| Variante | Type correct | Type top-1 | Type top-3 | Exact top-1 | Équivalent top-3 | Fonctionnel top-1 | Types testés | Secondes |
|---|---|---|---|---|---|---|---|---|
| `hier` (fragments, familles, types) | 0,41 | 0,44 | 0,52 | 0,37 | 0,46 | 0,34 | 20,1 | 0,38 |
| `flat_fragments` (mêmes fragments, sans hiérarchie) | 0,41 | 0,43 | 0,51 | 0,37 | 0,46 | 0,34 | 31,4 | 0,62 |
| `flat_scene` (tous les types sur la scène entière) | 0,40 | 0,42 | **0,63** | 0,33 | **0,56** | 0,30 | 16,6 | 0,51 |
| `hier` sans preuve négative | 0,41 | 0,44 | 0,51 | 0,37 | 0,46 | 0,34 | 20,1 | 0,38 |
| `hier`, preuve négative naïve | 0,38 | 0,42 | 0,49 | 0,36 | 0,44 | 0,33 | 20,1 | 0,38 |
| `hier` + boucle attribution ↔ type | 0,41 | 0,44 | 0,51 | 0,37 | 0,45 | 0,34 | 19,0 | 0,41 |

Type correct selon la part masquée (`hier`) : 0,82 à 0 % ; 0,59 à 20 % ; 0,41 à 40 % ; 0,21 à 60 % ; 0,04 à 80 %.
Pour `flat_scene` : 0,69 ; 0,50 ; 0,41 ; 0,29 ; 0,13. L'inférence sur fragments est meilleure quand la frontière est
visible, moins bonne quand elle est très masquée.

Type correct par type vrai (`hier`) : bande 0,65 ; disque 0,55 ; anneau 0,40 ; demi-plan 0,36 ; anti-disque 0,34 ;
secteur 0,16.

Confusions principales (type vrai → type prédit, sur 320 scènes par type) :

| Type vrai | Correct | Erreurs les plus fréquentes |
|---|---|---|
| bande | 207 | aucune hypothèse 43, disque 26, secteur 23 |
| disque | 176 | aucune hypothèse 72, anneau 32 |
| anneau | 129 | bande 67, disque 66, secteur 33 |
| demi-plan | 116 | aucune hypothèse 106, secteur 53 |
| anti-disque | 110 | aucune hypothèse 124, demi-plan 29 |
| secteur | 52 | aucune hypothèse 99, disque 74, demi-plan 66 |

Calibration : la confiance dépasse 0,8 dans 1 355 cas, pour une exactitude de 55 % ; erreur de calibration 0,34. Le
score n'est pas calibré.

Signaux, hypothèses correctes contre fausses : milieux expliqués de près 37,6 contre 26,6 ; part de frontière vue
0,53 contre 0,39 ; résidu de la contrainte partagée 0,06 contre 0,14. L'écart géométrique (0,09 contre 0,10), la part
absente (0,04 contre 0,05) et la stabilité (0,02 contre 0,02) ne séparent presque pas.

## 4. Matrice oracle : d'où vient la perte

Récupération exacte de la cible en première hypothèse, banc de masquage :

| | Type vrai | Type inféré |
|---|---|---|
| **Fragments vrais** | 0,66 | 0,50 |
| **Fragments estimés** | 0,51 | 0,34 |

Différences appariées, IC 95 % : attribution, à type vrai, −0,15 [−0,17 ; −0,13] ; type, à fragments vrais, −0,16
[−0,18 ; −0,15] ; les deux ensemble −0,32 [−0,34 ; −0,30]. Les deux pertes sont de même taille et s'additionnent.

Selon la part masquée :

| Masque | Fragments vrais, type vrai | Fragments estimés, type vrai | Fragments vrais, type inféré | Fragments estimés, type inféré |
|---|---|---|---|---|
| 0 % | 0,95 | 0,92 | 0,86 | 0,80 |
| 20 % | 0,88 | 0,73 | 0,77 | 0,51 |
| 40 % | 0,78 | 0,51 | 0,57 | 0,32 |
| 60 % | 0,53 | 0,32 | 0,25 | 0,08 |
| 80 % | 0,17 | 0,09 | 0,04 | 0,00 |

Même avec fragments et type vrais, il reste 34 % d'échecs : c'est la part des paramètres, nulle sans masque et
dominante à 80 %.

Même matrice dans la chaîne aval (structure candidate atteinte, 128 tâches) :

| Condition | Structure candidate | Accord final |
|---|---|---|
| vrais atomes | 0,59 | 0,956 |
| fragments vrais, type vrai, paramètres inférés | 0,51 | 0,929 |
| fragments vrais, type inféré | 0,44 | 0,911 |
| fragments estimés, type vrai (`po_type_k3`) | 0,35 | 0,901 |
| fragments estimés, type inféré (`hier_k3`) | 0,27 | 0,874 |
| pipeline gelé | 0,20 | 0,864 |

## 5. Propriété du masquage

La primitive vraie est donnée ; le système doit dire pourquoi ses portions manquantes manquent, avec ses propres
hypothèses comme candidats masques.

| Masque | Partie cachée étiquetée « cachée » | Partie cachée étiquetée « absente » | Partie visible étiquetée « observée » | Bon propriétaire trouvé |
|---|---|---|---|---|
| 20 % | 0,02 | 0,39 | 0,96 | 1 % |
| 40 % | 0,11 | 0,52 | 0,96 | 8 % |
| 60 % | 0,20 | 0,54 | 0,95 | 15 % |
| 80 % | 0,19 | 0,59 | 0,91 | 12 % |

Le reste de la partie cachée est classé hors domaine ou trou court. La partie visible n'est presque jamais prise
pour cachée (moins de 1 %). Mais le masque n'est reconnu comme propriétaire que dans 8 à 15 % des cas : la frontière
cachée reste majoritairement comptée comme absente.

## 6. Branche secteur (64 scènes par niveau)

| Masque | Exact | Demi-droites trouvées (sur 2) | Pris pour un disque | Étape en échec la plus fréquente |
|---|---|---|---|---|
| 0 % | 0,23 | 1,61 | 30 % | demi-droites 20, relation de sommet 14, position du sommet 13 |
| 20 % | 0,05 | 1,31 | 33 % | demi-droites 38 |
| 40 % | 0,05 | 1,14 | 47 % | demi-droites 44 |
| 60 % | 0,00 | 0,86 | 48 % | demi-droites 62 |
| 80 % | 0,00 | 0,39 | 80 % | demi-droites 63 |

Sans masque, l'échec se répartit sur trois étapes. Dès 20 % de masque, il est presque toujours en amont : les deux
demi-droites ne sont pas trouvées comme fragments droits, et les milieux du secteur sont lus comme un arc de cercle.

## 7. Mesure active pour lever une ambiguïté de type (192 scènes, 16 mesures)

| Politique | Type correct | Exact top-1 |
|---|---|---|
| aucune mesure | 0,45 | 0,48 |
| aléatoire | 0,49 | 0,45 |
| là où les types concurrents divergent | 0,44 | 0,47 |

Dirigée − aléatoire : type correct −0,06 [−0,13 ; +0,01].

## 8. Chaîne aval sans oracle de type (128 tâches de composition)

`attribution → type → paramètres → structure candidate → fonction`

| Bras | Oracle | F1 d'attribution | Rappel de type | Paramètres retrouvés | Littéraux | Structure candidate | Final ≥ 0,95 | Accord final | Secondes |
|---|---|---|---|---|---|---|---|---|---|
| pipeline gelé | aucun | — | 0,33 | 0,21 | 77,9 | 0,20 | 0,24 | 0,864 | 18,2 |
| découverte robuste | aucun | — | 0,30 | 0,26 | 3,6 | 0,22 | 0,30 | 0,693 | 5,6 |
| `hier_k1` | aucun | 0,51 | 0,23 | 0,17 | 2,7 | 0,22 | 0,23 | 0,865 | 11,5 |
| **`hier_k3`** (bras principal) | aucun | 0,51 | 0,36 | 0,26 | 7,1 | 0,27 | 0,30 | 0,874 | 12,5 |
| `hier_k3` + tours résiduels | aucun | 0,51 | 0,36 | 0,26 | 7,1 | 0,27 | 0,34 | 0,879 | 16,2 |
| `flat_scene_k1` | aucun | — | 0,33 | 0,26 | 7,0 | 0,27 | 0,32 | 0,882 | 11,4 |
| `flat_scene_k3` | aucun | — | 0,55 | 0,40 | 21,0 | **0,33** | **0,37** | **0,895** | 13,8 |
| `flat_scene_k3` + tours résiduels | aucun | — | 0,55 | 0,40 | 21,0 | 0,32 | 0,37 | 0,889 | 17,7 |
| `po_type_k3` | types | — | 0,55 | 0,40 | 9,3 | 0,35 | 0,44 | 0,901 | 12,0 |
| vrais atomes | tout | — | 1,00 | 1,00 | 3,5 | 0,59 | 0,75 | 0,956 | 8,0 |

Différences appariées contre le pipeline gelé, IC 95 % :

| Bras | Structure candidate | Final ≥ 0,95 | Accord final |
|---|---|---|---|
| `hier_k3` | +0,06 [−0,01 ; +0,13] | +0,06 [−0,01 ; +0,13] | +0,010 [−0,004 ; +0,025] |
| `hier_k3` + tours résiduels | +0,06 [0,00 ; +0,13] | +0,10 [+0,03 ; +0,17] | +0,016 [+0,003 ; +0,029] |
| `flat_scene_k1` | +0,07 [+0,02 ; +0,13] | +0,08 [+0,01 ; +0,15] | +0,019 [+0,006 ; +0,032] |
| `flat_scene_k3` | +0,13 [+0,07 ; +0,19] | +0,13 [+0,06 ; +0,20] | +0,032 [+0,018 ; +0,046] |
| `po_type_k3` (oracle de type) | +0,15 [+0,08 ; +0,22] | +0,20 [+0,13 ; +0,27] | +0,037 [+0,024 ; +0,051] |

Autres comparaisons : `flat_scene_k3` − `po_type_k3`, accord final −0,006 [−0,016 ; +0,005] ; `flat_scene_k3` −
`hier_k3`, accord final +0,021 [+0,010 ; +0,034] ; `hier_k3` − `hier_k1`, accord final +0,009 [−0,002 ; +0,021] et
final ≥ 0,95 +0,08 [+0,02 ; +0,13]. Le garde-fou ne change rien sur ces bras.

Par profondeur, structure candidate (`flat_scene_k3` + résiduel / pipeline gelé) : 0,88 / 0,63 à la profondeur 2 ;
0,25 / 0,13 à 3 ; 0,13 / 0,03 à 4 ; 0,03 / 0,03 à 5.

Feuilles dont le type est retrouvé (`flat_scene_k3` / `hier_k3` / pipeline gelé) : disque 0,61 / 0,51 / 0,36 ;
anti-disque 0,63 / 0,41 / 0,14 ; bande 0,55 / 0,45 / 0,32 ; demi-plan 0,48 / 0,20 / 0,55 ; anneau 0,39 / 0,16 / 0,19 ;
secteur 0,34 / 0,07 / 0,26. Secteurs dont les paramètres sont retrouvés : 0,04 / 0,00 / 0,08.

## 9. Hypothèses pré-enregistrées

| # | Résultat |
|---|---|
| FT1 attribution après type > proximité | **soutenue** : F1 +0,11 [+0,10 ; +0,12] |
| FT2 hiérarchie > classification plate sur fragments | **non soutenue** : +0,003 [−0,003 ; +0,008] |
| FT3 fragments > tous les types sur la scène | **non soutenue** : +0,010 [−0,015 ; +0,035] |
| FT4 top-3 − top-1 ≥ 0,10 | **non soutenue** pour `hier` (0,52 − 0,44 = 0,08) ; 0,21 pour `flat_scene` |
| FT5 preuve négative limitée aux zones observables | **non soutenue** : égale à l'absence de preuve négative (−0,002) ; meilleure que la version naïve (+0,027 [+0,015 ; +0,039]) |
| FT6 boucle attribution ↔ type | **non soutenue** : 0,000 [−0,004 ; +0,004] |
| FT7 frontière cachée reconnue (≥ 0,5 à 40 %) | **non soutenue** : 0,11 |
| FT8 secteurs récupérables (≥ 0,5 sans masque) | **non soutenue** : 0,23 |
| FT9 mesure active > aléatoire | **non soutenue** : −0,06 [−0,13 ; +0,01] |
| FT10 aval sans oracle : structure candidate > pipeline gelé | **non soutenue** pour `hier_k3` : +0,06 [−0,01 ; +0,13] |
| FT11 aval sans oracle : accord final > pipeline gelé | **non soutenue** pour `hier_k3` : +0,010 [−0,004 ; +0,025] |
| FT12 top-3 de types aide l'aval | **non soutenue** sur le critère (+0,009 [−0,002 ; +0,021]) ; final ≥ 0,95 +0,08 [+0,02 ; +0,13] |

Résultat exploratoire prévu par la clause du pré-enregistrement : un autre bras sans oracle, `flat_scene_k3`, bat le
pipeline gelé sur la structure candidate et sur l'accord final, avec des intervalles au-dessus de zéro. Il n'était
pas le bras principal ; il ne change pas le verdict.

## 10. Réponses

**1. Quelle part de perte vient de l'attribution des fragments ?**
Environ la moitié de ce qui se perd avant les paramètres : −0,15 [−0,17 ; −0,13] de récupération exacte sur le banc de
masquage, et −0,16 de structure candidate dans la chaîne (0,51 avec fragments vrais contre 0,35 avec fragments
estimés, type vrai dans les deux cas).

**2. Quelle part vient du type ?**
La même sur le banc de masquage : −0,16 [−0,18 ; −0,15] à fragments vrais. Dans la chaîne, elle est plus petite :
−0,07 de structure candidate à fragments vrais. Et elle disparaît si l'on renonce à choisir : essayer les six types
sur la scène et garder trois hypothèses par type donne exactement le rappel de type de l'oracle (0,55) et un accord
final équivalent (−0,006 [−0,016 ; +0,005]). Dans la chaîne, le type n'est donc pas un verrou propre ; l'attribution
en est un.

**3. Le top-k de types aide-t-il ?**
Oui, mais moins par groupe de fragments que par type sur la scène. Pour `hier`, passer de 1 à 3 types par groupe
fait gagner 0,08 de rappel de type et +0,08 de tâches à 0,95, sans effet net sur l'accord moyen. Pour `flat_scene`,
passer de 1 à 3 hypothèses par type fait passer le rappel de type de 0,33 à 0,55 et la structure candidate de 0,27
à 0,33.

**4. La hiérarchie de types aide-t-elle ?**
Non en exactitude : +0,003. Elle réduit seulement le calcul, de 31 à 20 types testés par scène et de 0,62 à 0,38 s.
La petite grammaire de cinq familles suffit à exprimer les six types, mais ne les distingue pas mieux qu'un
classement plat par score.

**5. Les secteurs deviennent-ils récupérables ?**
Non. 0,23 sans masque, 0,05 à 20 %, 0 au-delà ; c'est moins bien qu'à type connu (0,69 sans masque au sous-jalon
précédent). Les demi-droites ne sont pas isolées comme fragments droits et le secteur est lu comme un disque dans 30
à 80 % des cas. En composition, le type secteur est proposé pour 34 % des feuilles par `flat_scene_k3`, mais ses
paramètres ne sont retrouvés que pour 4 %.

**6. L'occlusion ownership aide-t-elle ?**
Non. Le mécanisme est prudent (il ne prend presque jamais une frontière visible pour cachée) mais il n'identifie le
masque que dans 8 à 15 % des cas, et son effet sur le type est nul par rapport à l'absence de preuve négative. Le seul
résultat net est négatif : compter toute frontière manquante comme absente dégrade le type (−0,027).

**7. Le gain traverse-t-il jusqu'à la structure finale ?**
Pour le bras principal, non : `hier_k3` améliore l'attribution mais reste dans l'incertitude sur la structure
candidate (+0,06 [−0,01 ; +0,13]) et l'accord final (+0,010 [−0,004 ; +0,025]). Pour `flat_scene_k3`, oui, sur toute
la chaîne : rappel de type +0,22, paramètres +0,19, structure candidate +0,13, accord final +0,032, tous avec un
intervalle au-dessus de zéro.

**8. Le coût reste-t-il compact ?**
Oui. Aucun ajustement par gradient dans l'inférence : 0,4 à 0,7 s par scène, 17 à 33 types testés, 11 fragments. En
chaîne, 12,5 à 13,8 s par tâche contre 18,2 s pour le pipeline gelé, avec 7 à 21 littéraux au lieu de 78. Toute la
campagne tient sur deux c6i.xlarge pendant environ une heure vingt.

**9. Quelle est la nouvelle erreur dominante ?**
L'attribution sous composition. Dans la chaîne, le F1 d'attribution n'est que de 0,51 ; avec les fragments vrais, la
structure candidate passe de 0,35 à 0,51 à type égal. Sur le banc, 60 % des groupes par modèle mêlent cible et
masque. Derrière elle : la profondeur (structure candidate 0,13 à la profondeur 4 pour `flat_scene_k3` avec tours
résiduels) et la
récupération à 0,99, qui reste sous 5 % pour tous les bras sans oracle.

**10. Peut-on enfin construire une chaîne sans oracle de type supérieure au pipeline gelé ?**
Oui, sur ces 128 tâches de dev, mais pas avec le bras que j'avais désigné. `flat_scene_k3` n'utilise aucun oracle et
bat le pipeline gelé : structure candidate 0,33 contre 0,20 (+0,13 [+0,07 ; +0,19]), accord final 0,895 contre 0,864
(+0,032 [+0,018 ; +0,046]), avec presque quatre fois moins de littéraux et en moins de temps. C'est un résultat exploratoire, sur un bras
de comparaison, à confirmer avant toute conclusion.

## 11. Limites

- Dev uniquement, une graine par scène, aucune confirmation.
- Le bras qui bat le pipeline gelé n'était pas le bras principal pré-enregistré. Une douzaine de bras sans oracle ont été
  comparés à la référence ; le meilleur est exposé à un biais de sélection.
- `flat_scene` n'utilise ni fragments explicites ni attribution : sa réussite ne valide pas l'approche par fragments
  de ce sous-jalon, elle montre que l'inférence de paramètres du sous-jalon précédent se passe d'oracle de type.
- Le banc de masquage n'a qu'une forme de composition (`cible ∧ masque`).
- Score de type non calibré ; rôles des fragments (`CompatibleFragmentAssignments`) construits et testés mais non
  évalués quantitativement.
- La polarité d'un fragment est lue localement ; sous composition elle peut être fausse (anti-disque : 0,34).
- Aucun modèle appris n'a été essayé : les approches explicites plafonnent, mais le plafond vient de l'attribution,
  qu'un classifieur de type ne corrigerait pas.

## 12. Verdict

Règle pré-enregistrée : FT1 soutenue ; FT10 et FT11 non soutenues pour le bras principal.

`FRAGMENT/TYPE INFERENCE IMPROVED BUT INSUFFICIENT`

L'attribution et le type sont maintenant séparés et mesurés : ils pèsent autant l'un que l'autre sur le banc de
masquage, et l'attribution domine en composition. L'inférence explicite par fragments n'apporte pas de gain aval
établi. En revanche, essayer tous les types sans attribution donne une chaîne sans oracle au-dessus du pipeline gelé ;
ce résultat est exploratoire. Le pipeline gelé reste la référence ; rien de ce sous-jalon n'y est branché.

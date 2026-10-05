# WCE Partial-Observation Primitive Parameter Inference — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucun test confirmatoire, aucune campagne 16 graines,
lot scellé jamais touché. Le pipeline gelé reste la référence ; rien de ce sous-jalon n'y est branché par défaut.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur les scènes de la sous-plage dev + 350 000 (18 à 36 scènes par condition) et sur la
tâche de composition 599. Évaluation sur des scènes jamais vues (sous-plage dev + 300 000 : 6 types × 5 niveaux de
masquage × 2 niveaux de bruit × 32 primitives = 1 920 scènes, appariées par primitive) et sur 128 tâches de
composition jamais utilisées (clés 500–515).

Méthode principale : `arc_visible` (consensus local, estimateur d'arc géométrique, contraintes partagées, score
tolérant au masquage). Référence : `global_contrast` (optimisation globale par contraste, celle du plafond « vrais
types » du sous-jalon précédent). En aval : `po_type_k3` contre `types_old` et contre le pipeline gelé.

| # | Hypothèse | Critère |
|---|---|---|
| PO1 | l'inférence sur fragments récupère plus de primitives que l'optimisation globale, type connu | `recovered_1`, différence appariée, IC 95 % > 0 |
| PO2 | le gain persiste sous masquage ≥ 40 % | idem, sous-ensemble masqué |
| PO3 | l'estimateur géométrique bat Kåsa sur fragments partiels | `arc_geometric` − `arc_kasa`, IC > 0 |
| PO4 | les contraintes partagées aident les primitives riches | `arc_joint` − `arc_geometric`, sous-ensemble anneau/bande/secteur, IC > 0 |
| PO5 | le score tolérant au masquage aide | `arc_visible` − `arc_joint`, IC > 0 |
| PO6 | le contexte de composition (masque inféré, rendu contrefactuel) améliore le top-1 | `ctx_joint` − `arc_visible`, IC > 0 |
| PO7 | le top-5 contient la vérité nettement plus souvent que le top-1 | `recovered_5` − `recovered_1` ≥ 0,10 |
| PO8 | les régions de paramètres couvrent la vérité dans ≥ 80 % des cas | couverture |
| PO9 | les mesures actives dirigées vers les paramètres battent l'aléatoire | différence appariée, IC > 0 |
| PO10 | aval, type connu : structure candidate au-dessus de l'ancienne estimation | `po_type_k3` − `types_old`, IC > 0 |
| PO11 | aval : accord final au-dessus du pipeline gelé | `po_type_k3` − `reference` sur `final_full`, IC > 0 |
| PO12 | l'échange primitive ↔ composition améliore l'accord final | `loop_full` − `final_full`, IC > 0 |

Attente déclarée d'après la mise au point : PO1, PO2, PO7 probables ; PO6 probable à 40 % de masquage ; secteurs
attendus faibles ; PO10–PO12 inconnues.

Règle de verdict fixée d'avance :

- `PARAMETER INFERENCE READY FOR END-TO-END VALIDATION` : PO1, PO10 et PO11 soutenues.
- `PARAMETER INFERENCE IMPROVED BUT INSUFFICIENT` : PO1 soutenue, mais PO10 ou PO11 non soutenue.
- `PARAMETER INFERENCE STILL BLOCKING` : PO1 non soutenue.

## 1. Protocole exécuté

- Run AWS `wce-deeppo-20261005-005310` (EC2 + SSM, 2 × c6i.xlarge, 57 et 60 min, 3 072 sous-unités, 0 erreur,
  29 568 lignes). Commandes : `make deep-run-po`, `make deep-fetch`, `make deep-po-aggregate`, `make deep-po-report`.
  Résumé : `results/deep_partial_observation_summary.json`.
- Masquage contrôlé : une primitive vraie (`T`), cachée par un masque (`T ∧ M`) qui retire 0, 20, 40, 60 ou 80 % de sa
  frontière dans la zone observée ; masque demi-plan pour les primitives circulaires, disque pour les rectilignes.
  Bruit d'étiquette 0 ou 8 %. 300 observations. 1 920 scènes, appariées par primitive. Le TYPE est fourni, pas les
  paramètres.
- Chaîne aval : 128 tâches de composition jamais utilisées, type de chaque feuille fourni.
- Code : `core/deep_partial.py` (système, sans vérité), `core/deep_partial_exp.py` (évaluation),
  `core/deep_po_report.py`. Tests : 73 nouveaux (`make deep-po-test`) ; suite complète 4 783 tests, 7 ignorés ; lot scellé
  intact.
- Note de vocabulaire : dans la grammaire WCE un secteur est un angle non borné (sommet, direction, demi-ouverture) ;
  il n'a pas de rayon. Ses paramètres analysés sont donc le sommet et les deux angles.
- Aucun seuil n'a été réglé sur l'évaluation. Une seule graine par scène ; pas de test confirmatoire.

## 2. Récupération de la primitive, type connu (1 920 scènes)

`top-1` : la première hypothèse a toutes ses frontières dans la tolérance de la métrique d'équivalence.

| Méthode | top-1 | top-3 | top-5 | Accord fonctionnel top-1 | Secondes |
|---|---|---|---|---|---|
| `global_contrast` (optimisation globale, référence) | 0,31 | 0,34 | 0,34 | 0,73 | 0,11 |
| `global_target` (gradient direct sur la cible) | 0,35 | 0,37 | 0,37 | 0,81 | 0,07 |
| `arc_normals` (intersection des normales) | 0,39 | 0,46 | 0,48 | 0,79 | 0,12 |
| `arc_kasa` | 0,44 | 0,52 | 0,55 | 0,81 | 0,13 |
| `arc_taubin` | 0,43 | 0,51 | 0,54 | 0,80 | 0,13 |
| `arc_geometric` | 0,43 | 0,51 | 0,54 | 0,80 | 0,14 |
| `arc_joint` (+ contraintes partagées) | 0,51 | 0,61 | 0,65 | 0,84 | 0,22 |
| **`arc_visible`** (+ score tolérant au masquage) | 0,49 | 0,61 | 0,64 | 0,84 | 0,21 |
| `ctx_joint` (structure connue, masque inféré) | 0,56 | 0,63 | 0,65 | 0,87 | 0,36 |
| `ctx_oracle` (masque vrai fourni) | 0,62 | 0,64 | 0,64 | 0,89 | 0,21 |

top-1 selon la part de frontière masquée :

| Méthode | 0 % | 20 % | 40 % | 60 % | 80 % |
|---|---|---|---|---|---|
| `global_contrast` | 0,52 | 0,46 | 0,32 | 0,19 | 0,07 |
| `global_target` | **0,92** | 0,60 | 0,23 | 0,02 | 0,01 |
| `arc_visible` | 0,90 | 0,74 | 0,46 | 0,28 | 0,08 |
| `ctx_joint` | 0,96 | 0,80 | 0,60 | 0,35 | 0,11 |
| `ctx_oracle` | 0,96 | 0,88 | 0,67 | 0,43 | 0,14 |
| `arc_visible`, top-5 | 0,97 | 0,90 | 0,70 | 0,46 | 0,18 |

top-1 de `arc_visible` par type (entre parenthèses : `global_target`) :

| Type | 0 % | 20 % | 40 % | 60 % | 80 % |
|---|---|---|---|---|---|
| demi-plan | 0,98 (1,00) | 0,86 (0,80) | 0,67 (0,00) | 0,39 (0,00) | 0,11 (0,00) |
| bande | 0,97 (1,00) | 0,92 (0,92) | 0,86 (0,45) | 0,67 (0,06) | 0,17 (0,03) |
| disque | 0,92 (1,00) | 0,97 (1,00) | 0,41 (0,59) | 0,14 (0,00) | 0,02 (0,00) |
| anti-disque | 0,92 (0,97) | 0,59 (0,02) | 0,22 (0,00) | 0,11 (0,00) | 0,00 (0,00) |
| anneau | 0,91 (0,92) | 0,78 (0,59) | 0,58 (0,28) | 0,39 (0,03) | 0,19 (0,02) |
| secteur | 0,69 (0,64) | 0,31 (0,27) | 0,05 (0,03) | 0,00 (0,00) | 0,00 (0,00) |

Bruit et masquage séparés (`arc_visible`, top-1) :

| | masque 0 % | 40 % | 80 % |
|---|---|---|---|
| sans bruit | 0,95 | 0,50 | 0,08 |
| bruit 8 % | 0,84 | 0,43 | 0,08 |

Pour `global_contrast` : 0,67 sans bruit ni masque, 0,36 avec bruit seul, 0,48 avec 40 % de masque seul. Le bruit est
le premier problème de l'optimisation globale ; le masquage est le premier problème de l'inférence sur fragments.

Différences appariées (IC 95 % bootstrap, top-1) :

| Comparaison | Toutes scènes | Masque ≥ 40 % | Primitives riches |
|---|---|---|---|
| `arc_visible` − `global_contrast` | +0,18 [+0,16 ; +0,20] | +0,08 [+0,06 ; +0,11] | +0,25 [+0,22 ; +0,28] |
| `arc_visible` − `global_target` | +0,14 [+0,12 ; +0,16] | +0,19 [+0,17 ; +0,22] | +0,15 [+0,12 ; +0,18] |
| `arc_geometric` − `arc_kasa` | −0,01 [−0,02 ; −0,00] | −0,01 [−0,02 ; +0,00] | −0,01 [−0,01 ; +0,00] |
| `arc_joint` − `arc_geometric` | +0,07 [+0,06 ; +0,09] | +0,07 [+0,06 ; +0,09] | +0,15 [+0,12 ; +0,17] |
| `arc_visible` − `arc_joint` | −0,01 [−0,03 ; −0,00] | 0,00 [−0,02 ; +0,02] | −0,03 [−0,05 ; −0,01] |
| `ctx_joint` − `arc_visible` | +0,07 [+0,05 ; +0,09] | +0,08 [+0,05 ; +0,10] | +0,06 [+0,04 ; +0,08] |
| `ctx_oracle` − `arc_visible` | +0,12 [+0,11 ; +0,14] | +0,14 [+0,12 ; +0,16] | +0,09 [+0,07 ; +0,11] |

## 3. Paramètres

Part des estimations top-1 hors tolérance (`arc_visible`, tous niveaux de masquage) :

| Type | Paramètre | Hors tolérance |
|---|---|---|
| secteur | sommet | 70 % |
| secteur | direction | 62 % |
| secteur | demi-ouverture | 54 % |
| anti-disque | centre | 62 % |
| anti-disque | rayon | 54 % |
| disque | centre | 48 % |
| disque | rayon | 35 % |
| anneau | centre | 38 % |
| anneau | rayon extérieur | 30 % |
| anneau | rayon intérieur | 22 % |
| demi-plan | orientation | 34 % |
| demi-plan | décalage | 32 % |
| bande | largeur / décalage / orientation | 20 % / 19 % / 18 % |

Erreur médiane du disque : centre 0,08 et rayon 0,03 sans masque ; 0,38 et 0,18 à 40 % ; 1,44 et 0,63 à 80 %.
À 40 % de masque, l'anneau garde un centre à 0,18 et la bande une orientation à 0,03 rad.

Oracles partiels sur le banc de masquage (top-1, différence appariée avec `arc_visible`) :

| Paramètre vrai fourni | Toutes scènes applicables | Masque ≥ 40 % |
|---|---|---|
| orientation (demi-plan, bande, secteur) | +0,20 [+0,17 ; +0,22] | +0,26 [+0,22 ; +0,30] |
| centre (disque, anti-disque, anneau, secteur) | +0,21 [+0,18 ; +0,24] | +0,33 [+0,29 ; +0,38] |
| taille (rayons, largeur, demi-ouverture) | +0,11 [+0,09 ; +0,13] | +0,15 [+0,12 ; +0,18] |

Pour anneaux et secteurs, fournir le centre n'aide pas (−0,05 [−0,09 ; +0,00]) : la recherche des rayons ou des angles
par modes d'histogramme autour du centre est moins bonne que l'ajustement des fragments.

### Régions de paramètres, identifiabilité, sensibilité locale (cercles et droites simples)

| Masque | Arc observé (rad) | Conditionnement médian | Corrélation rayon–centre | Écart-type postérieur du rayon | Largeur moyenne | Couverture de la vérité |
|---|---|---|---|---|---|---|
| 0 % | 4,71 | 3,6 | 0,40 | 0,02 | 0,27 | 0,87 |
| 20 % | 3,99 | 5,3 | 0,56 | 0,04 | 0,37 | 0,65 |
| 40 % | 3,11 | 15,9 | 0,75 | 0,09 | 0,55 | 0,42 |
| 60 % | 2,54 | 54,2 | 0,83 | 0,17 | 0,84 | 0,30 |
| 80 % | 2,00 | 297,9 | 0,89 | 0,26 | 1,10 | 0,22 |

Le diagnostic local se comporte comme attendu : quand l'arc raccourcit, une direction devient plate et rayon et
centre deviennent presque interchangeables. Mais la région ne couvre la vérité que dans 49 % des cas en moyenne : elle
est construite autour du fragment attribué à la première hypothèse, et quand ce fragment mêle la frontière de la
primitive et celle du masque, la région est centrée au mauvais endroit.

Étiquettes d'identifiabilité : un paramètre étiqueté `IDENTIFIABLE` est dans la tolérance dans 72 % des cas,
`PARTIALLY_IDENTIFIABLE` dans 21 %, `UNIDENTIFIABLE` dans 18 %. L'ordre est juste, la calibration ne l'est pas.

Complétion de frontière : la part de frontière complétée qui tombe sur la vraie frontière passe de 0,92 (sans masque)
à 0,57 (40 %) puis 0,30 (80 %). La fraction visible estimée est correcte jusqu'à 40 % (0,60 pour 0,60), puis
surestimée (0,49 pour 0,20 à 80 %).

## 4. Contraintes partagées, base minimale

Rappel de TOUTES les composantes de frontière parmi 5 hypothèses, primitive riche avec contraintes partagées contre
composantes minimales indépendantes :

| Type | 0 % | 20 % | 40 % | 60 % | 80 % |
|---|---|---|---|---|---|
| anneau, riche | 0,95 | 0,92 | 0,78 | 0,55 | 0,27 |
| anneau, minimal | 0,50 | 0,34 | 0,14 | 0,05 | 0,00 |
| bande, riche | 1,00 | 0,98 | 0,95 | 0,88 | 0,33 |
| bande, minimal | 0,95 | 0,89 | 0,67 | 0,33 | 0,08 |
| secteur, riche | 0,84 | 0,61 | 0,20 | 0,05 | 0,02 |
| secteur, minimal | 0,70 | 0,48 | 0,16 | 0,02 | 0,00 |

Pour disque, anti-disque et demi-plan les deux bases sont identiques par construction.

## 5. Autopsie des secteurs (`arc_visible`, 64 scènes par niveau)

| Masque | top-1 | top-5 | Demi-droites trouvées (sur 2) | Plus courte demi-droite visible | Échec : sommet | Échec : angles | Le vrai secteur passerait le seuil calibré |
|---|---|---|---|---|---|---|---|
| 0 % | 0,69 | 0,84 | 1,81 | 4,50 | 17 sur 20 | 0 | 88 % |
| 20 % | 0,31 | 0,61 | 1,63 | 2,78 | 38 sur 44 | 1 | 61 % |
| 40 % | 0,05 | 0,20 | 1,11 | 1,16 | 51 sur 61 | 4 | 53 % |
| 60 % | 0,00 | 0,05 | 0,80 | 0,29 | 51 sur 64 | 12 | 36 % |
| 80 % | 0,00 | 0,02 | 0,47 | 0,13 | 61 sur 64 | 2 | 11 % |

L'échec dominant est le sommet : il est l'intersection de deux droites, et une petite erreur d'angle sur une
demi-droite courte le déplace beaucoup. Dès 40 % de masque, une des deux demi-droites n'a plus qu'environ 1,2 unité
visible et n'est plus trouvée qu'une fois sur deux. La validation trop stricte est un facteur secondaire : sans
masque, 12 % des vrais secteurs seraient déjà rejetés par le seuil calibré du sous-jalon précédent ; à 60 %, 64 %.

## 6. Mesures actives dirigées vers les paramètres (192 scènes, masque 40 ou 60 %, bruit 5 %, 16 mesures)

| Politique | top-1 | top-5 | Accord fonctionnel |
|---|---|---|---|
| aucune mesure | 0,43 | 0,61 | 0,84 |
| aléatoire | 0,43 | 0,59 | 0,82 |
| désaccord entre hypothèses | 0,49 | 0,66 | 0,85 |
| désaccord, zones masquées exclues | 0,47 | 0,65 | 0,85 |

Désaccord − aléatoire : top-1 +0,06 [−0,01 ; +0,13] ; accord fonctionnel +0,03 [+0,01 ; +0,05].

## 7. Chaîne aval (128 tâches de composition, type de chaque feuille fourni sauf mention)

| Littéraux | Feuilles retrouvées (top-1 / top-k) | Équivalence | Littéraux | Structure candidate ≥ 0,97 | Final ≥ 0,95 | Final ≥ 0,99 | Accord final | Après échange |
|---|---|---|---|---|---|---|---|---|
| vrais atomes | — | 1,00 | 3,5 | 0,67 | 0,77 | 0,11 | 0,963 | 0,970 |
| vrais atomes + garde-fou | — | 1,00 | 3,5 | 0,94 | 0,88 | 0,65 | 0,980 | 0,980 |
| ancienne estimation, centres et orientations vrais | — | 0,41 | 3,5 | 0,44 | 0,53 | 0,11 | 0,921 | 0,921 |
| ancienne estimation, type seul (`types_old`) | — | 0,20 | 3,5 | 0,21 | 0,28 | 0,06 | 0,877 | 0,877 |
| **nouvelle inférence, type seul, k = 1** | 0,26 / 0,26 | 0,31 | 3,3 | 0,33 | 0,39 | 0,10 | 0,899 | 0,899 |
| **nouvelle inférence, type seul, k = 3** | 0,26 / 0,42 | 0,49 | 9,8 | 0,41 | 0,52 | 0,10 | 0,914 | 0,914 |
| nouvelle inférence, type seul, k = 5 | 0,26 / 0,49 | 0,55 | 16,2 | 0,37 | 0,51 | 0,09 | 0,915 | 0,915 |
| k = 3, score global au lieu de visible | 0,30 / 0,44 | 0,51 | 9,8 | 0,39 | 0,49 | 0,10 | 0,915 | 0,915 |
| k = 3 + garde-fou | 0,26 / 0,42 | 0,49 | 9,8 | 0,42 | 0,49 | 0,10 | 0,912 | 0,912 |
| nouvelle inférence + centre vrai, k = 1 | 0,34 | 0,40 | 3,5 | 0,38 | 0,44 | 0,10 | 0,896 | 0,896 |
| nouvelle inférence + orientation vraie, k = 1 | 0,39 | 0,43 | 3,9 | 0,39 | 0,48 | 0,10 | 0,913 | 0,913 |
| nouvelle inférence + taille vraie, k = 1 | 0,38 | 0,44 | 3,8 | 0,41 | 0,50 | 0,09 | 0,910 | 0,910 |
| pipeline gelé, aucun oracle (référence) | — | 0,46 | 77,1 | 0,21 | 0,26 | 0,03 | 0,876 | 0,876 |
| découverte robuste, aucun oracle | — | 0,23 | 3,2 | 0,24 | 0,32 | 0,07 | 0,654 | 0,653 |

Différences appariées par tâche, IC 95 % :

| Comparaison | Équivalence des primitives | Structure candidate | Accord final |
|---|---|---|---|
| k = 3 − `types_old` | +0,29 [+0,23 ; +0,34] | +0,20 [+0,13 ; +0,27] | +0,037 [+0,025 ; +0,050] |
| k = 1 − `types_old` | +0,11 [+0,05 ; +0,16] | +0,12 [+0,04 ; +0,20] | +0,022 [+0,009 ; +0,035] |
| k = 3 − k = 1 | +0,18 [+0,15 ; +0,22] | +0,08 [+0,02 ; +0,14] | +0,015 [+0,004 ; +0,026] |
| k = 3 − pipeline gelé | +0,03 [−0,03 ; +0,08] | +0,20 [+0,13 ; +0,27] | +0,038 [+0,025 ; +0,051] |
| vrais atomes − k = 1 | +0,69 [+0,65 ; +0,73] | +0,34 [+0,27 ; +0,43] | +0,063 [+0,049 ; +0,078] |
| échange − sans échange, k = 3 | — | — | 0,000 [−0,000 ; +0,000] |
| échange − sans échange, vrais atomes | — | — | +0,008 [+0,006 ; +0,010] |

Feuilles retrouvées en composition (k = 3, top-1 / top-3) : demi-plan 0,31 / 0,56 ; bande 0,30 / 0,50 ;
disque 0,23 / 0,39 ; anti-disque 0,27 / 0,50 ; anneau 0,31 / 0,38 ; secteur 0,06 / 0,07.

Par profondeur, équivalence des primitives (k = 3) : 0,64 (profondeur 2), 0,56 (3), 0,45 (4), 0,31 (5).

Coût : la nouvelle inférence ne fait aucun ajustement par gradient (0,21 s par primitive, 4 hypothèses en moyenne).
En chaîne, k = 3 coûte 12,8 s par tâche contre 19,0 s pour le pipeline gelé et 11,8 s pour l'ancienne estimation.
AWS : 2 × c6i.xlarge pendant une heure pour toute la campagne.

## 8. Hypothèses pré-enregistrées

| # | Résultat |
|---|---|
| PO1 inférence sur fragments > optimisation globale | **soutenue** : +0,18 [+0,16 ; +0,20] |
| PO2 gain sous masque ≥ 40 % | **soutenue** : +0,08 [+0,06 ; +0,11] |
| PO3 estimateur géométrique > Kåsa | **non soutenue** : −0,01 [−0,02 ; −0,00] |
| PO4 contraintes partagées, primitives riches | **soutenue** : +0,15 [+0,12 ; +0,17] |
| PO5 score tolérant au masquage | **non soutenue** : −0,01 [−0,03 ; −0,00] |
| PO6 contexte de composition | **soutenue** : +0,07 [+0,05 ; +0,09] |
| PO7 top-5 − top-1 ≥ 0,10 | **soutenue** : 0,64 − 0,49 = 0,15 |
| PO8 couverture des régions ≥ 80 % | **non soutenue** : 0,87 sans masque, 0,49 en moyenne |
| PO9 mesures actives > aléatoire | **non soutenue** sur le critère (top-1 +0,06 [−0,01 ; +0,13]) ; accord fonctionnel +0,03 [+0,01 ; +0,05] |
| PO10 aval : structure candidate > ancienne estimation | **soutenue** : +0,20 [+0,13 ; +0,27] |
| PO11 aval : accord final > pipeline gelé | **soutenue** : +0,038 [+0,025 ; +0,051] — mais le bras reçoit les types, le pipeline gelé non |
| PO12 échange primitive ↔ composition | **non soutenue** : 0,000 ; +0,008 seulement avec les vrais atomes |

## 9. Réponses

**1. Quelle fraction de frontière faut-il pour estimer correctement chaque primitive ?**
Sans bruit, pour que la première hypothèse soit correcte au moins une fois sur deux : bande et anneau 40 % de
frontière visible ; demi-plan 60 % ; disque et anti-disque 80 % ; secteur 100 % (et encore : 0,69). Avec cinq
hypothèses : bande, anneau et demi-plan 40 % ; disque et anti-disque 60 % ; secteur 80 %. Les primitives à deux
frontières liées tiennent mieux que les primitives à une seule : la seconde frontière fixe le centre ou la direction.

**2. Quels paramètres sont les plus fragiles ?**
Le sommet du secteur (hors tolérance dans 70 % des cas), puis ses angles ; le centre des cercles (48 à 62 %), toujours
plus fragile que le rayon ; l'orientation et le décalage d'un demi-plan isolé (un tiers). Les plus solides sont ceux
de la bande (moins de 20 %). Fournir le centre ou l'orientation vrais vaut +0,20 de récupération, et +0,26 à +0,33
sous masque fort ; fournir la taille vaut moitié moins.

**3. L'occlusion explicite améliore-t-elle le fit ?**
Oui quand elle passe par le contexte, non quand elle passe par le score seul. Le score « tolérant au masquage », qui
ignore les tronçons muets, ne fait pas mieux que le contraste global (−0,01). Rendre le composite entier et ne
comparer que ce qui doit être visible apporte +0,07 avec un masque inféré et +0,12 avec le masque vrai. L'écart entre
les deux (0,05) est ce que coûte l'erreur sur le masque lui-même.

**4. Les contraintes partagées aident-elles ?**
Oui, c'est le gain le plus net du sous-jalon : +0,15 [+0,12 ; +0,17] sur les primitives riches. À 60 % de masque,
l'anneau avec centre partagé est retrouvé dans 55 % des cas contre 5 % pour deux cercles indépendants, la bande dans
88 % contre 33 %. Pour le secteur, le gain existe mais reste faible (5 % contre 2 %).

**5. Top-k paramétrique améliore-t-il la composition ?**
Oui jusqu'à k = 3 : structure candidate 0,33 → 0,41 (+0,08 [+0,02 ; +0,14]), accord final +0,015. Non au-delà :
k = 5 retrouve plus de feuilles (0,49 contre 0,42) mais redescend à 0,37 de structure candidate, avec 16 littéraux.

**6. Le raisonnement compositionnel peut-il raffiner les primitives ?**
Sur le banc de masquage, oui : connaître la structure et choisir conjointement primitive et masque donne +0,07. Dans
la chaîne complète, non : l'échange borné après composition ne change rien (0,000), sauf avec les vrais atomes
(+0,008 d'accord, et la récupération à 0,99 passe de 11 % à 34 %). L'échange affine des paramètres déjà proches ; il
ne répare pas une structure construite sur de mauvais paramètres.

**7. Les secteurs restent-ils le principal échec ?**
Oui. 0,69 sans masque, 0,05 à 40 %, 0,00 au-delà ; 6 % des secteurs retrouvés en composition. L'échec est le sommet
dans 80 à 95 % des cas, parce qu'une des deux demi-droites devient trop courte pour être trouvée.

**8. Les mesures actives deviennent-elles utiles pour les paramètres ?**
Pas de façon établie. Mesurer là où les hypothèses divergent donne top-1 0,49 contre 0,43 pour l'aléatoire, mais
l'intervalle inclut zéro ; seul l'accord fonctionnel progresse nettement (+0,03). Exclure les zones masquées n'ajoute
rien. Elles ne sont pas intégrées.

**9. Le gain paramétrique améliore-t-il réellement le downstream ?**
Oui, à type connu, et sur toute la chaîne : feuilles retrouvées, équivalence des primitives 0,20 → 0,49, structure
candidate 0,21 → 0,41, accord final 0,877 → 0,914, tous avec un intervalle au-dessus de zéro. Trois réserves : le
type de chaque feuille est fourni par oracle ; la récupération à 0,99 reste à 10 % ; l'écart aux vrais atomes reste
grand (0,41 contre 0,67 de structure candidate).

**10. Quel est le prochain goulot ?**
Deux, dans l'ordre. D'abord l'attribution des fragments : sous composition, un fragment mêle souvent la frontière de
la primitive et celle de ce qui la masque, et tout le reste (région, identifiabilité, complétion) est alors centré au
mauvais endroit ; c'est ce que montrent la couverture des régions (49 %) et le gain du masque vrai (+0,12). Ensuite
l'absence de chaîne sans oracle de type : la découverte robuste sans oracle reste à 0,65 d'accord final, et rien ici
ne dit quel type proposer pour quel fragment.

## 10. Limites

- Dev uniquement, une graine par scène, aucune confirmation.
- Le banc de masquage n'a qu'une forme de composition (`T ∧ M`) et un masque d'un autre type géométrique que la cible.
- Tous les gains aval supposent le type de chaque feuille connu. La comparaison au pipeline gelé (PO11) n'est donc
  pas à armes égales.
- La métrique top-1 a des tolérances fixes ; l'accord fonctionnel bouge moins que la récupération.
- L'estimateur géométrique, le score tolérant au masquage et l'échange en chaîne n'apportent rien de mesurable ;
  ils restent dans le code comme bras de comparaison.
- Régions de paramètres et étiquettes d'identifiabilité ne sont calculées que pour cercles et droites simples.
- Déterminisme : à graine fixée les hypothèses sont identiques à la précision machine près (écarts de 2e-16 d'un
  appel à l'autre, dus à l'algèbre linéaire multi-fil) ; le test de déterminisme compare donc à 1e-9.
- Formalisation `PrimitiveNode` / `CompositeNode` : seule la visibilité circule du composite vers la primitive ;
  aucune contrainte géométrique n'est déduite de la structure.

## 11. Verdict

Règle pré-enregistrée : PO1, PO10 et PO11 sont soutenues, ce qui donne littéralement
`PARAMETER INFERENCE READY FOR END-TO-END VALIDATION`.

Je ne retiens pas ce verdict, et je le signale comme un **écart déclaré à la règle, dans le sens conservateur**. La
règle que j'ai écrite ne prévoyait pas que PO11 compare un bras recevant les types par oracle à un pipeline qui n'en
reçoit aucun. Une validation de bout en bout n'a pas d'oracle de type, et la seule chaîne sans oracle mesurée ici est
moins bonne que le pipeline gelé.

`PARAMETER INFERENCE IMPROVED BUT INSUFFICIENT`

L'inférence des paramètres à type connu est nettement meilleure et le gain traverse toute la chaîne. Elle reste
insuffisante au-delà de 40 % de masque, sur les secteurs, et tant qu'aucune chaîne sans oracle n'en bénéficie. Le
pipeline gelé reste la référence ; rien de ce sous-jalon n'y est branché.

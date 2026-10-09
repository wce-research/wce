# WCE Boundary Ownership & Fragment Attribution — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucun test confirmatoire, aucune campagne 16 graines,
lot scellé jamais touché. Le pipeline gelé reste la référence ; rien de ce sous-jalon n'y est branché par défaut.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur 120 scènes de la sous-plage dev + 350 000 et sur la tâche de composition 799.
Réglages figés avant le lancement (`results/deep_configs/ownership.json`) : coût par paramètre 10, terme « absent » à
zéro, coût de hasard 3, température 16.

Évaluation sur 1 920 scènes de masquage jamais vues (indices 200–231 de la sous-plage dev + 300 000) et sur **256
tâches de composition jamais utilisées** (clés 700–731).

### Confirmation de `flat_scene_k3` (comparaison principale, fixée ici)

Au sous-jalon précédent, `flat_scene_k3` a battu le pipeline gelé, mais il avait été choisi parmi une douzaine de
bras. La confirmation porte sur **trois bras seulement, désignés d'avance** : pipeline gelé, `flat_scene_k3`, et la
méthode d'appartenance proposée `own_k3`. Jeu de tâches nouveau et indépendant, deux fois plus grand.

Critère : `flat_scene_k3` − pipeline gelé, différence appariée par tâche, IC 95 % au-dessus de zéro **à la fois** sur
la structure candidate et sur l'accord final. Si le critère est rempli, `flat_scene_k3` devient une **baseline
candidate** ; sinon, il reste un résultat exploratoire non confirmé. Ce statut est indépendant du verdict du
sous-jalon.

### Hypothèses

| # | Hypothèse | Critère |
|---|---|---|
| BO1 | l'attribution globale bat les micro-fragments locaux | F1 par paires, `solver` − `tangent`, IC 95 % > 0 |
| BO2 | le recollement réduit le morcellement | morceaux par vraie frontière, `solver` − `tangent`, IC < 0 |
| BO3 | l'association longue distance aide | F1, `solver` − `solver_local`, IC > 0 |
| BO4 | le modèle d'occlusion aide | appartenance top-1 : `solver_absent` − `solver` > 0 et `solver_absent` − `solver_absent_naive` > 0 |
| BO5 | le top-3 d'appartenance contient nettement plus souvent le bon propriétaire | top-3 − top-1 ≥ 0,10 |
| BO6 | la composition aide à ré-attribuer | chaîne : F1 après composition − F1 du solveur, IC > 0 |
| BO7 | le score d'appartenance est calibré | erreur de calibration ≤ 0,10 sur l'évaluation |
| BO8 | la visibilité contrefactuelle bat le contraste | récupération top-1, masques inférés, IC > 0 |
| BO9 | avec appartenance et type vrais, un secteur non masqué est retrouvé | récupération ≥ 0,5 ; sinon le paramétrage est intrinsèquement fragile |
| BO10 | `flat_scene_k3` se confirme | critère ci-dessus |
| BO11 | la méthode d'appartenance bat le pipeline gelé | `own_k3` − référence, structure candidate et accord final, IC > 0 |
| BO12 | la méthode d'appartenance bat `flat_scene_k3` | `own_k3` − `flat_scene_k3`, accord final, IC > 0 |

Attente déclarée d'après la mise au point : BO1, BO2, BO3 probables ; BO4 probablement fausse (le terme « absent »
dégrade l'attribution en mise au point, avec ou sans masque explicite) ; BO7 incertaine ; BO10 probable ; BO11 et BO12
peu probables.

Limite connue avant lancement : aucun coût de hasard testé ne ramène sous 10 % la part des milieux attribués sur des
étiquettes aléatoires. Le plus fort (3) laisse 32 % ; sans ce coût, 96 %. Le bras sans coût de hasard
(`solver_no_chance`, `own_nc_k3`) est rapporté à part, comme comparaison.

Les bras de comparaison non désignés (`own_nc_k3`, `flat_owned_k3`, `flat_own_union_k3`, `po_owned_k3`) et la matrice
oracle sont exploratoires et ne changent ni le verdict ni le statut de `flat_scene_k3`.

Règle de verdict fixée d'avance :

- `BOUNDARY OWNERSHIP READY FOR END-TO-END VALIDATION` : BO1, BO11 et BO12 soutenues.
- `BOUNDARY OWNERSHIP IMPROVED BUT INSUFFICIENT` : BO1 soutenue, mais BO11 ou BO12 non soutenue.
- `BOUNDARY OWNERSHIP STILL BLOCKING` : BO1 non soutenue.

## 1. Protocole exécuté

- Run AWS `wce-deepbo-20261005-211529` (EC2 + SSM, 2 × c6i.xlarge, 80 et 85 min, 3 296 sous-unités, 0 erreur,
  26 176 lignes). Commandes : `make deep-run-bo`, `make deep-fetch`, `make deep-bo-aggregate`, `make deep-bo-report`.
  Résumé : `results/deep_boundary_ownership_summary.json`.
- Banc de masquage : 1 920 scènes jamais vues. Chaîne aval : 256 tâches de composition jamais utilisées.
- Code : `core/deep_ownership.py` (système, sans vérité, sans modèle appris), `core/deep_ownership_exp.py`
  (évaluation), `core/deep_bo_report.py`. Tests : 54 nouveaux (`make deep-bo-test`) ; suite complète 4 894 tests OK,
  7 ignorés ; lot scellé intact.
- Réglages figés avant le lancement ; aucun seuil réglé sur l'évaluation. Une seule graine par scène.
- Écart au protocole : aucun.

## 2. Confirmation de `flat_scene_k3` (256 tâches nouvelles, trois bras désignés)

| Bras | Primitives retrouvées | Littéraux | Structure candidate | Final ≥ 0,95 | Accord final | Secondes |
|---|---|---|---|---|---|---|
| pipeline gelé | 0,26 | 80,3 | 0,19 | 0,21 | 0,881 | 18,2 |
| **`flat_scene_k3`** | 0,40 | 20,9 | **0,32** | **0,37** | **0,904** | 13,7 |
| `own_k3` (méthode d'appartenance) | 0,04 | 2,1 | 0,13 | 0,16 | 0,620 | 4,4 |

`flat_scene_k3` − pipeline gelé, différences appariées, IC 95 % :

- structure candidate : **+0,137 [+0,090 ; +0,184]** ;
- accord final : **+0,023 [+0,015 ; +0,031]** ;
- final ≥ 0,95 : +0,160 [+0,109 ; +0,215] ; primitives retrouvées : +0,148 [+0,106 ; +0,189].

Le critère pré-enregistré est rempli sur ses deux volets. **`flat_scene_k3` devient une baseline candidate.** Il reste
à type inconnu, sans oracle, avec quatre fois moins de littéraux et un quart de temps en moins. Le gain d'accord final
est plus petit qu'au sous-jalon précédent (+0,023 contre +0,032), ce qui est attendu d'un bras sélectionné puis
re-mesuré. La récupération à 0,99 reste à 6 %.

Par profondeur, structure candidate (`flat_scene_k3` / pipeline gelé) : 0,70 / 0,48 à 2 ; 0,44 / 0,17 à 3 ;
0,13 / 0,06 à 4 ; 0,03 / 0,03 à 5.

## 3. Attribution (1 920 scènes)

| Méthode | Précision | Rappel | F1 | Pureté | Morceaux par vraie frontière | Longueur moyenne | Appartenance top-1 | top-3 |
|---|---|---|---|---|---|---|---|---|
| micro-fragments (`tangent`) | 0,88 | 0,46 | 0,57 | 0,94 | 2,58 | 0,63 | — | — |
| recollement local (`shared`) | 0,80 | 0,58 | 0,64 | 0,91 | 2,15 | 1,14 | — | — |
| consensus | 0,65 | 0,83 | 0,71 | 0,81 | 1,50 | 2,09 | — | — |
| **`solver`** (global, calibré par le hasard) | 0,72 | 0,78 | 0,73 | 0,80 | 1,13 | 2,69 | 0,39 | 0,41 |
| `solver_no_chance` | 0,76 | 0,81 | **0,77** | 0,84 | 1,38 | 2,67 | **0,48** | 0,53 |
| `solver_local` (sans longue distance) | 0,82 | 0,55 | 0,62 | 0,88 | 1,21 | 1,35 | 0,32 | 0,36 |
| `solver_no_smooth` | 0,70 | 0,76 | 0,71 | 0,78 | 1,09 | 2,82 | 0,37 | 0,39 |
| `solver_absent` (preuve négative, masques explicites) | 0,56 | 0,56 | 0,53 | 0,79 | 0,78 | 2,44 | 0,27 | 0,28 |
| `solver_absent_naive` | 0,55 | 0,56 | 0,53 | 0,77 | 0,71 | 2,56 | 0,26 | 0,26 |
| composition (ré-attribution par un petit composite) | 0,67 | 0,74 | 0,66 | 0,77 | 0,96 | 2,82 | — | — |

Différences appariées, IC 95 % :

| Comparaison | F1 | Pureté | Morceaux par frontière | Appartenance top-1 |
|---|---|---|---|---|
| `solver` − `tangent` | +0,154 [+0,142 ; +0,165] | −0,138 [−0,145 ; −0,130] | −1,45 [−1,50 ; −1,39] | — |
| `solver` − consensus | +0,021 [+0,012 ; +0,029] | −0,008 [−0,014 ; −0,001] | −0,37 [−0,40 ; −0,34] | — |
| `solver` − `solver_local` | +0,113 [+0,102 ; +0,125] | −0,079 | −0,08 | +0,070 [+0,053 ; +0,088] |
| `solver` − `solver_no_smooth` | +0,023 [+0,018 ; +0,027] | +0,022 | +0,04 | +0,020 [+0,013 ; +0,027] |
| `solver` − `solver_no_chance` | −0,038 [−0,045 ; −0,030] | −0,031 | −0,25 | −0,085 [−0,098 ; −0,072] |
| `solver_absent` − `solver` | −0,197 [−0,211 ; −0,183] | −0,022 | −0,35 | −0,127 [−0,142 ; −0,112] |
| `solver_absent` − `solver_absent_naive` | +0,006 [+0,003 ; +0,009] | +0,014 | +0,06 | +0,006 [+0,002 ; +0,011] |
| composition − `solver` | −0,064 [−0,071 ; −0,057] | −0,036 | −0,17 | — |

Appartenance top-1 selon la part masquée : `solver` 0,69 ; 0,44 ; 0,30 ; 0,22 ; 0,31. `solver_no_chance` 0,81 ; 0,56 ;
0,41 ; 0,29 ; 0,33. Par type (`solver_no_chance`) : anti-disque 0,61 ; disque 0,54 ; anneau 0,49 ; demi-plan 0,48 ;
bande 0,46 ; secteur 0,29.

Énergie moyenne de la solution retenue, par terme : ajustement 25,8 ; milieux sans propriétaire 36,7 ; lissage 0,9 ;
complexité 36,8 ; absence 0 (terme désactivé) ; hasard 80,0. Le coût de hasard est le terme dominant : il décide
du nombre de courbes gardées (1,5 en moyenne, contre 2,0 sans lui). 27 attributions testées par scène, 0,21 s.

Détection du masque : un masque est déclaré dans 11 à 24 % des scènes, et il n'est le bon que dans 2 à 5 %. Sans
masque, il en est déclaré un à tort dans 11 % des scènes.

Calibration du score d'appartenance : erreur de calibration 0,094, score de Brier 0,25. Mais la confiance est presque
constante (0,29 à 0,54) pour une exactitude de 0,42 : le score est juste en moyenne et n'ordonne rien. Par niveau de
masque, l'erreur va de 0,08 à 0,22. Conséquence de la température choisie, 95 % des fragments sont classés ambigus.

## 4. Matrice oracle

Récupération exacte de la cible sur le banc de masquage. L'appartenance estimée sert de filtre : seules les hypothèses
dont toutes les frontières sont des courbes possédées sont gardées.

| Appartenance | Type vrai | Type inféré |
|---|---|---|
| vraie | 0,67 | 0,44 |
| aucune (scène entière) | 0,54 | 0,35 |
| estimée, comme filtre | 0,26 | 0,22 |
| estimée, primitives construites sur les courbes possédées | — | 0,25 |

Différences appariées : appartenance vraie − aucune, +0,13 [+0,11 ; +0,15] à type vrai et +0,08 [+0,07 ; +0,10] à type
inféré ; appartenance estimée − aucune, **−0,27 [−0,29 ; −0,25]** à type vrai et −0,13 [−0,16 ; −0,11] à type inféré.
Le filtre ne garde que 14 % des hypothèses, et la cible n'a toutes ses frontières possédées que dans 34 % des scènes.

Dans la chaîne aval (256 tâches) :

| Condition | Primitives retrouvées | Structure candidate | Accord final |
|---|---|---|---|
| vrais atomes | 1,00 | 0,65 | 0,959 |
| appartenance vraie, type vrai | 0,74 | 0,52 | 0,938 |
| appartenance vraie, type inféré | 0,65 | 0,40 | 0,919 |
| aucune attribution, type vrai (`po_type_k3`) | 0,40 | 0,39 | 0,915 |
| aucune attribution, type inféré (`flat_scene_k3`) | 0,40 | 0,32 | 0,904 |
| appartenance estimée sans coût de hasard (`own_nc_k3`) | 0,25 | 0,29 | 0,893 |
| appartenance estimée (`own_k3`) | 0,04 | 0,13 | 0,620 |

De `flat_scene_k3` (0,32) aux vrais atomes (0,65), l'écart de 0,33 se répartit ainsi : +0,08 en donnant l'appartenance
vraie, +0,12 en donnant ensuite le type, +0,13 en donnant enfin les paramètres.

## 5. Visibilité contrefactuelle contre contraste

Même vivier de cinq hypothèses à type vrai, quatre classements (récupération en première position) :

| Classement | Toutes scènes | Masque 40 % | Masque 60 % |
|---|---|---|---|
| contraste global | 0,55 | 0,59 | 0,30 |
| visibilité, sans masque supposé | 0,51 | 0,54 | 0,23 |
| visibilité, masques inférés | 0,52 | 0,55 | 0,23 |
| visibilité, masque vrai | 0,58 | 0,64 | 0,34 |

Contre le contraste : masques inférés −0,031 [−0,045 ; −0,017] ; masque vrai +0,032 [+0,018 ; +0,046]. Le principe
fonctionne quand le masque est connu ; il est inutilisable tant que le masque n'est pas trouvé. Le vivier ne contient
la vérité que dans 67 % des cas.

## 6. Secteurs : attribution ou paramétrage ?

| Masque | Appartenance et type vrais | Aucune attribution, type vrai | Appartenance top-1 des demi-droites | Milieux sur la demi-droite la plus pauvre | Erreur de sommet (appartenance vraie) |
|---|---|---|---|---|---|
| 0 % | 0,70 | 0,72 | 0,45 | 13,0 | 0,37 |
| 20 % | 0,55 | 0,25 | 0,31 | 8,5 | 0,73 |
| 40 % | 0,27 | 0,17 | 0,28 | 4,2 | 0,80 |
| 60 % | 0,02 | 0,09 | 0,32 | 1,1 | 1,54 |
| 80 % | 0,00 | 0,03 | 0,05 | 0,3 | 1,83 |

Avec appartenance et type vrais, les échecs sont le sommet dans la grande majorité des cas (15 sur 19 sans masque,
27 sur 47 à 40 %), puis l'absence d'hypothèse quand une demi-droite n'a plus de milieux.

## 7. Chaîne aval, bras exploratoires (256 tâches)

| Bras | Statut | F1 d'attribution | Appartenance top-1 | Littéraux | Structure candidate | Accord final |
|---|---|---|---|---|---|---|
| pipeline gelé | désigné | — | — | 80,3 | 0,19 | 0,881 |
| `flat_scene_k3` | désigné | — | — | 20,9 | 0,32 | 0,904 |
| `own_k3` | désigné | 0,30 | 0,09 | 2,1 | 0,13 | 0,620 |
| `own_nc_k3` | exploratoire | 0,57 | 0,38 | 7,0 | 0,29 | 0,893 |
| `flat_owned_k3` (hypothèses filtrées par appartenance) | exploratoire | 0,30 | 0,09 | 12,6 | 0,22 | 0,886 |
| `flat_own_union_k3` | exploratoire | 0,30 | 0,09 | 22,9 | 0,31 | 0,903 |
| `po_owned_k3` (oracle de type, filtré) | exploratoire | 0,30 | 0,09 | 7,5 | 0,30 | 0,904 |

Contre le pipeline gelé : `own_k3` −0,261 [−0,307 ; −0,214] d'accord final ; `own_nc_k3` +0,012 [+0,004 ; +0,020] et
+0,098 [+0,043 ; +0,152] de structure candidate. Contre `flat_scene_k3` : `own_k3` −0,284 [−0,331 ; −0,236] ;
`own_nc_k3` −0,011 [−0,018 ; −0,004] ; filtrage par appartenance −0,018 [−0,024 ; −0,012] ; union 0,000.

En composition, le solveur calibré par le hasard ne garde presque aucune courbe : 2,1 littéraux par tâche, 4 % de
primitives retrouvées. Les frontières y sont courtes et nombreuses, et aucune ne dépasse nettement ce qu'une courbe
capterait par hasard.

Ré-attribution par la composition (bras `own_k3`) : après composition, les milieux sont ré-attribués aux feuilles du
meilleur composite là où elles sont visibles. F1 0,30 → 0,53, soit +0,11 [+0,08 ; +0,14] sur les tâches appariées ;
appartenance top-1 0,09 → 0,32, soit +0,19 [+0,15 ; +0,23].

## 8. Hypothèses pré-enregistrées

| # | Résultat |
|---|---|
| BO1 attribution globale > micro-fragments | **soutenue** : F1 +0,154 [+0,142 ; +0,165] |
| BO2 le recollement réduit le morcellement | **soutenue** : 2,58 → 1,13 morceaux, −1,45 [−1,50 ; −1,39] ; pureté 0,94 → 0,80 |
| BO3 association longue distance | **soutenue** : F1 +0,113 [+0,102 ; +0,125] |
| BO4 modèle d'occlusion | **non soutenue** : le terme d'absence dégrade l'appartenance (−0,127 [−0,142 ; −0,112]) ; masques explicites contre version naïve +0,006 |
| BO5 top-3 − top-1 ≥ 0,10 | **non soutenue** : 0,41 − 0,39 = 0,02 (0,05 sans coût de hasard) |
| BO6 la composition aide à ré-attribuer | **soutenue** dans la chaîne (+0,11 [+0,08 ; +0,14]), en partant d'une attribution faible ; l'inverse sur le banc de masquage (−0,064) |
| BO7 calibration ≤ 0,10 | **soutenue sur le critère** (0,094), sans pouvoir de discrimination |
| BO8 visibilité contrefactuelle > contraste | **non soutenue** : −0,031 [−0,045 ; −0,017] avec masques inférés |
| BO9 secteur non masqué, appartenance et type vrais ≥ 0,5 | **soutenue** : 0,70 |
| BO10 confirmation de `flat_scene_k3` | **soutenue** : +0,137 et +0,023, intervalles au-dessus de zéro |
| BO11 `own_k3` > pipeline gelé | **non soutenue** : −0,261 d'accord final |
| BO12 `own_k3` > `flat_scene_k3` | **non soutenue** : −0,284 |

## 9. Réponses

**1. Quelle part de l'erreur vient réellement de l'ownership ?**
Moins que je ne l'avais écrit au sous-jalon précédent. Sur le banc de masquage, donner l'appartenance vraie vaut +0,13
[+0,11 ; +0,15] de récupération exacte. Dans la chaîne, sur l'écart de 0,33 de structure candidate entre
`flat_scene_k3` et les vrais atomes, l'appartenance vraie rend 0,08, le type 0,12 et les paramètres 0,13 : environ un
quart. Le précédent rapport lui attribuait 0,16 parce qu'il la mesurait à type vrai ; à type inféré, elle pèse moitié
moins. L'appartenance est un verrou parmi trois de taille comparable, pas le verrou dominant.

**2. Top-k ownership aide-t-il ?**
Non. Le bon propriétaire est dans les trois premiers à peine plus souvent qu'en premier (0,41 contre 0,39 ; 0,53
contre 0,48 sans coût de hasard). Quand le solveur se trompe, la bonne courbe n'est généralement pas parmi celles
qu'il a retenues : garder plusieurs propriétaires ne peut pas la faire revenir.

**3. Le stitching réduit-il la fragmentation ?**
Oui. Une vraie frontière passe de 2,6 morceaux à 1,1 et la longueur moyenne d'un groupe de 0,6 à 2,7. Le prix est la
pureté, de 0,94 à 0,80. Le recollement local seul ne suffit pas (2,2 morceaux).

**4. Les associations longue distance aident-elles ?**
Oui : F1 +0,11 [+0,10 ; +0,12], appartenance top-1 +0,07 [+0,05 ; +0,09]. Sans elles, le rappel tombe de 0,78 à 0,55.

**5. Le modèle d'occlusion aide-t-il ?**
Non. Le masque n'est correctement identifié que dans 2 à 5 % des scènes. Pénaliser la frontière attendue et absente
dégrade l'attribution, avec ou sans masques explicites. Le score de visibilité contrefactuelle perd contre le
contraste avec les masques inférés (−0,031) et ne gagne qu'avec le masque vrai (+0,032). L'idée est juste, sa
condition préalable — trouver le masque — n'est pas remplie.

**6. La composition aide-t-elle à réattribuer ?**
Dans la chaîne, oui, nettement : F1 d'attribution +0,11 et appartenance top-1 +0,19. Mais le point de départ est
l'attribution très faible du solveur calibré (0,30), et le niveau atteint (0,53) reste sous celui du solveur sans
coût de hasard (0,57). Sur le banc de masquage, où l'attribution de départ est meilleure, la composition la dégrade
(−0,064). Elle répare une mauvaise attribution, elle n'améliore pas une attribution correcte.

**7. Les secteurs restent-ils problématiques ?**
Oui, et la cause est maintenant séparée. Sans masque, appartenance et type vrais donnent 0,70, autant que sans
aucune attribution (0,72) : le paramétrage n'est pas intrinsèquement fragile et l'attribution n'est pas le problème.
À 20 % de masque, l'appartenance vraie vaut beaucoup (0,55 contre 0,25) : là, c'est l'attribution. À partir de 40 %,
même l'appartenance vraie échoue (0,27, puis 0,02) : la demi-droite la plus pauvre n'a plus que quatre milieux, puis
un, et le sommet n'est plus déterminé. C'est un manque de données, pas un défaut de méthode.

**8. flat_scene_k3 se confirme-t-il sur nouveau dev ?**
Oui. Sur 256 tâches nouvelles, avec trois bras désignés d'avance : structure candidate +0,137 [+0,090 ; +0,184],
accord final +0,023 [+0,015 ; +0,031]. Il devient une baseline candidate. Ce n'est pas une validation confirmatoire :
dev uniquement, une graine.

**9. Le gain ownership traverse-t-il jusqu'à la structure finale ?**
Non. L'attribution s'améliore (F1 0,57 → 0,73), mais la méthode d'appartenance désignée fait bien pire en aval que
le pipeline gelé (accord 0,62 contre 0,88). Sa variante sans coût de hasard bat le pipeline gelé (+0,012) mais reste
sous `flat_scene_k3` (−0,011). Utiliser l'appartenance estimée comme filtre fait perdre 0,27 de récupération sur le
banc et 0,018 d'accord dans la chaîne. Aucun usage de l'appartenance estimée ne fait mieux que de ne rien attribuer.

**10. Quel est le nouveau goulot ?**
Il n'y en a plus un seul en amont. Entre `flat_scene_k3` et les vrais atomes, appartenance, type et paramètres se
partagent l'écart à parts voisines, et aucune des trois n'a cédé à une méthode géométrique explicite. Le plafond
lui-même est bas : avec les vrais atomes, la structure candidate n'est atteinte que dans 65 % des tâches (38 % à la
profondeur 5) et la récupération à 0,99 dans 10 %. Le prochain goulot à isoler est donc en aval de l'amont : la
composition en profondeur et l'ajustement final, à atomes donnés.

## 10. Limites

- Dev uniquement, une graine par scène et par tâche. La confirmation de `flat_scene_k3` est une re-mesure sur données
  nouvelles, pas un test confirmatoire.
- La calibration par le hasard n'atteint pas sa cible : en mise au point, 32 % des milieux restent attribués sur des
  étiquettes aléatoires (96 % sans le coût). Elle n'a pas été re-mesurée sur l'évaluation.
- Le bras désigné `own_k3` porte ce coût de hasard, qui l'effondre en composition. Le bras sans ce coût est meilleur
  en aval mais n'est pas calibré : ni l'un ni l'autre n'est satisfaisant.
- Le score d'appartenance est calibré en moyenne par une température élevée qui l'aplatit ; il ne sépare pas les
  attributions justes des fausses.
- Le banc de masquage n'a qu'une forme de composition (`cible ∧ masque`).
- Solveur glouton avec au plus trois reprises ; ni appariement optimal ni programmation en nombres entiers n'ont été
  essayés. Le résultat négatif porte sur ce solveur et cette énergie.
- Les rôles « limite du domaine » et « inconnu » des hypothèses d'appartenance sont construits et testés, pas
  évalués quantitativement.

## 11. Verdict

Règle pré-enregistrée : BO1 soutenue ; BO11 et BO12 non soutenues.

`BOUNDARY OWNERSHIP IMPROVED BUT INSUFFICIENT`

L'attribution mesurée progresse (recollement, longue distance), mais elle ne produit aucun gain aval et, utilisée
comme filtre, elle nuit. Le modèle d'occlusion échoue faute de trouver le masque. Résultat séparé, acquis :
`flat_scene_k3` est confirmé sur données nouvelles et devient une baseline candidate. Le pipeline gelé reste la
référence officielle ; rien de ce sous-jalon n'y est branché.

Statut enregistré dans `results/deep_configs/baselines.json` : `flat_scene_k3` = `candidate baseline` (gain confirmé sur
un nouveau jeu de dev ; ne remplace pas la baseline officielle confirmatoire). `own_k3` n'est pas branché.

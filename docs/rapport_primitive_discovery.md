# Rapport — WCE Primitive Discovery

Sous-jalon EXPLORATOIRE du jalon « Deep Compositional Structure Discovery & Model-Class Boundary ».
Données : partition **dev uniquement** (banc atomique et scènes d'entraînement sur des sous-plages dédiées de dev).
Ni la validation, ni le test confirmatoire, ni le lot scellé n'ont été touchés. Rien ici n'est confirmatoire : tout est
`POST-HOC / EXPLORATORY` au sens du pré-enregistrement.

Sources : `results/deep_primitive_summary.json` (run AWS `wce-deepprim-20261004-150003`, 1 536 sous-unités, aucune en
erreur), `results/deep_neural_training.json` (run `wce-deepneural-20261004-145510`).
Cibles : `make deep-train-proposer`, `deep-install-proposer`, `deep-run-prim`, `deep-prim-aggregate`,
`deep-prim-report`.

Effectifs : 144 scènes atomiques par méthode (9 catégories × 16), soit 248 atomes vrais ; 8 scènes par décalage hors
distribution ; 8 scènes par politique active ; 8 tâches par cellule (profondeur × régime) pour la chaîne aval.

## Résumé

La question du sous-jalon était : peut-on découvrir assez bien les primitives pour rendre les structures profondes
accessibles à la composition ? **Réponse mesurée : pas encore.**

- Aucun des nouveaux proposeurs ne dépasse l'ajustement par gradient direct, qui reste la meilleure méthode du banc
  atomique (équivalence 0,56 contre 0,46 à 0,49 pour les méthodes géométrique et hybrides).
- Le proposeur neuronal compact n'apporte aucun gain de rappel : seul, il est le plus faible (0,22) ; en hybride, il
  égale l'hybride sans réseau (0,49). Il est en revanche le plus précis (0,54) et le plus rapide.
- La chaîne causale est vérifiée : quand toutes les primitives sont retrouvées, une structure candidate fonctionnelle
  apparaît dans 56 à 100 % des cas, contre 8 à 23 % sinon. Le manque d'atomes explique bien l'échec de génération.
- Un second goulot apparaît en aval : même avec les VRAIS atomes, la récupération finale au seuil 0,99 n'est atteinte
  que dans 0 à 25 % des cas, alors que l'accord moyen est de 0,85 à 0,98.

## 1. Architecture

`core/deep_primitives.py` — distinctions maintenues dans le code : `Primitive` (élément de base) ≠ `Fragment`
(composition réutilisable) ≠ `CompositeStructure` (assemblage).

- `PrimitiveCandidate` : objet explicite — type, paramètres, domaine, support observationnel, incertitude, résidu
  expliqué, coût descriptif, invariants, parenté. Jamais un vecteur latent.
- `PrimitiveProposal` (interface) : `GradientProposer` (référence), `GeometricProposer` (droites et cercles par
  consensus, aucun ajustement), `ContrastProposer` (gradient sur un objectif de contraste), `RefiningProposer`
  (hybride : les propositions des autres servent d'initialisation), `NeuralProposer` (réseau compact).
- `PrimitiveValidation` : le validateur WCE. Signal explicite, indépendant de la polarité : parmi les paires de points
  voisins que la frontière du candidat sépare, la part de paires d'étiquettes opposées doit dépasser le taux de base
  (écart z). Seuil corrigé des comparaisons multiples (√(2·ln N) + 1). Critères : support, contraste, stabilité au
  rééchantillonnage, équivalence avec une primitive retenue.
- `CompatiblePrimitives` : hypothèses multiples pondérées, regroupées par classes d'équivalence (accord ou complément
  ≥ 0,97 sur la région observée) ; le type d'une primitive peut rester ouvert.
- `discover` : boucle guidée par le résidu (proposer → valider → retenir → retirer les paires expliquées → reproposer),
  arrêt quand plus aucun candidat ne passe la validation. Le résidu est un ensemble de paires inexpliquées, pas une
  mémoire de points.
- `PrimitiveFamily`, `PrimitiveLibrary`, `BirthProposal`, `merge_redundant` : familles paramétriques, bibliothèque,
  naissance, fusion, retrait.
- `core/deep_primitive_eval.py` (évaluation) : métriques et classes d'identifiabilité. `core/deep_primitive_world.py`
  (évaluation) : banc atomique.

## 2. Résultats du banc atomique

| Méthode | Exact | Fonctionnel | Équivalence | Composantes | Précision | Erreur de paramètres | Classes retenues | Propositions | Équiv.-ajustement | Temps |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient direct (référence) | 0,34 | 0,47 | **0,56** | 0,60 | 0,37 | 0,24 | 2,1 | 76 | 38 | 1,0 s |
| Géométrique analytique | 0,29 | 0,38 | 0,46 | 0,53 | 0,25 | 0,20 | 4,3 | 159 | 0 | 0,15 s |
| Contraste | 0,05 | 0,23 | 0,34 | 0,39 | 0,12 | 0,42 | 4,4 | 105 | 63 | 2,3 s |
| Hybride | 0,27 | 0,38 | 0,49 | 0,56 | 0,19 | 0,28 | 4,6 | 301 | 67 | 3,4 s |
| Hybride, base minimale | 0,28 | 0,41 | 0,47 | 0,56 | 0,27 | 0,07 | 4,6 | 245 | 50 | 1,4 s |
| Neuronal compact | 0,12 | 0,17 | 0,22 | 0,26 | **0,54** | 0,12 | 0,8 | 36 | 0 | 0,03 s |
| Neuronal + hybride | 0,28 | 0,38 | 0,49 | 0,56 | 0,20 | 0,29 | 4,5 | 368 | 65 | 3,4 s |

Définitions : *exact* = même type et accord ≥ 0,95 ; *fonctionnel* = accord ≥ 0,95 ou complément, tout type ;
*équivalence* = toutes les composantes de frontière de l'atome (droites, cercles, demi-droites) retrouvées ;
*précision* = part des classes retenues dont toutes les composantes correspondent à une frontière vraie.

Lecture : moins de six atomes sur dix sont retrouvés même par la meilleure méthode, et entre la moitié et près de neuf
dixièmes des primitives retenues sont fausses. Compacité : les méthodes hybrides retiennent 3 classes par atome vrai
(1,4 pour le gradient, 0,5 pour le réseau).

### Par type d'atome (hybride)

| Atome | n | Exact | Équivalence | Composantes |
|---|---|---|---|---|
| Demi-plan | 43 | 0,28 | 0,93 | 0,93 |
| Bande | 25 | 0,32 | 0,64 | 0,80 |
| Disque | 103 | 0,42 | 0,53 | 0,53 |
| Anti-disque | 25 | 0,04 | 0,32 | 0,32 |
| Anneau | 31 | 0,03 | 0,03 | 0,26 |
| Secteur | 21 | 0,05 | 0,05 | 0,29 |

Les droites sont bien retrouvées ; les cercles à moitié ; les anneaux et les secteurs presque jamais. La reconnaissance
des macros (deux cercles concentriques → anneau, deux demi-droites → secteur) ne fonctionne pas.

### Par catégorie (équivalence)

| Catégorie | Gradient | Géométrique | Hybride | Neuronal | Neuronal + hybride |
|---|---|---|---|---|---|
| Un atome | 0,63 | 0,44 | 0,50 | 0,25 | 0,44 |
| Deux atomes | 0,28 | 0,44 | 0,38 | 0,16 | 0,38 |
| Plusieurs atomes | 0,17 | 0,36 | 0,34 | 0,08 | 0,44 |
| Atomes proches | 0,25 | 0,56 | 0,56 | 0,25 | 0,53 |
| Atomes imbriqués | 0,72 | 0,56 | 0,59 | 0,25 | 0,66 |
| Atomes redondants | 0,97 | 0,75 | 0,81 | 0,38 | 0,81 |
| Bruit élevé | 0,63 | 0,38 | 0,50 | 0,13 | 0,38 |
| Outliers groupés | 0,88 | 0,56 | 0,63 | 0,31 | 0,69 |
| Frontière partielle | 0,50 | 0,06 | 0,06 | 0,19 | 0,06 |

Le gradient direct domine sur les scènes simples (un atome, redondants, bruit, outliers, frontière partielle) ; les
méthodes géométriques et hybrides font mieux dès qu'il y a plusieurs atomes (deux, plusieurs, proches). Aucune méthode
n'est bonne partout. Les frontières partiellement observées mettent les méthodes géométriques en échec (0,06).

## 3. Identifiabilité

Classement des atomes vrais AVANT de juger (contraste du vrai atome sur les observations) : 244 identifiables, 4
partiels, 0 non identifiable. Le banc atomique ne contient donc presque que des atomes identifiables : les rappels
ci-dessus ne sont pas pénalisés par des atomes impossibles à voir. Équivalence sur les seuls identifiables : 0,56
(gradient), 0,49 (hybride), 0,50 (neuronal + hybride), 0,22 (neuronal). Sur les 4 partiels : 0 à 0,25.

Conséquence : l'échec de découverte n'est pas un problème d'identifiabilité des atomes. Le signal existe (les vrais
atomes ont un contraste élevé), mais ni les proposeurs ne le trouvent de façon fiable, ni le validateur ne l'isole des
fausses primitives.

## 4. Proposeur neuronal compact

| | |
|---|---|
| Rôle | propose des cercles (centre, rayon) et des droites (ancre, direction) ; ne prédit ni structure ni étiquettes |
| Paramètres | 23 881 |
| Mémoire | 95,5 Ko |
| Coût d'un passage | 13,7 millions de multiplications-additions |
| Entraînement | 4 000 scènes de dev, 30 époques, 158 s sur `c6i.xlarge` |
| Perte | 3,28 → 1,54 (curriculum complet) ; 3,04 → 0,99 (atomes seuls) |
| Gain réel sur le rappel des primitives | **aucun** : 0,22 seul ; 0,49 en hybride, comme l'hybride sans réseau |

Le réseau est précis mais muet : il propose peu (36 candidats), 0,8 classe retenue par scène. Sa validation finale reste
côté WCE.

### Généralisation hors distribution (scènes à deux atomes, équivalence)

| Décalage | Hybride (sans apprentissage) | Neuronal, curriculum complet | Neuronal, atomes seuls | Neuronal + hybride |
|---|---|---|---|---|
| Référence nominale | 0,38 | 0,16 | non mesuré | 0,38 |
| Positions nouvelles | 0,25 | 0,00 | 0,06 | 0,25 |
| Tailles nouvelles | 0,50 | 0,06 | 0,06 | 0,50 |
| Orientations nouvelles | 0,69 | 0,19 | 0,31 | 0,63 |
| Densités différentes | 0,56 | 0,06 | 0,19 | 0,56 |
| Bruit différent | 0,19 | 0,06 | 0,13 | 0,19 |

Le réseau ne généralise pas : son rappel tombe à 0,00–0,06 sur des positions, tailles, densités ou bruits nouveaux.

### Curriculum

Le réseau entraîné sur le curriculum complet (atomes → deux atomes → scènes multiples → compositions profondes) ne fait
pas mieux, hors distribution, que celui entraîné sur des atomes seuls (0,00–0,19 contre 0,06–0,31). La comparaison en
distribution nominale n'a pas été lancée. L'effet du curriculum sur les structures profondes n'est donc pas établi, et
le peu qui est mesuré ne le soutient pas.

## 5. Base riche, minimale, apprise

| | Base riche (8 atomes) | Base minimale (demi-plan, disque, anti-disque) |
|---|---|---|
| Équivalence | 0,49 | 0,47 |
| Précision | 0,19 | 0,27 |
| Erreur de paramètres | 0,28 | 0,07 |
| Temps | 3,4 s | 1,4 s |
| Accord final en aval, riche d2 / d3 | 0,978 / 0,930 | 0,969 / 0,925 |
| Accord final en aval, standard d2 / d3 | 0,929 / 0,868 | 0,949 / 0,921 |

La base minimale fait jeu égal en rappel, mieux en précision et en erreur de paramètres, pour 2,4 fois moins de temps.
Rien n'indique que les atomes redondants (bande, anneau, secteur) aident la découverte. Longueur des structures : en
base minimale une structure vraie compte 1,5 fois plus d'atomes (mesuré au sous-jalon précédent). La base « apprise »
n'a pas été évaluée comme base de travail : seules des propositions de naissance ont été produites (section 8).

## 6. Mesures actives dirigées vers les primitives

16 mesures supplémentaires, puis redécouverte (hybride).

| Scènes | Sans mesure | 16 mesures aléatoires | 16 mesures dirigées |
|---|---|---|---|
| Deux atomes — équivalence | 0,31 | 0,56 | 0,56 |
| Deux atomes — précision | 0,12 | 0,27 | 0,27 |
| Plusieurs atomes — équivalence | 0,28 | 0,47 | 0,47 |
| Plusieurs atomes — précision | 0,31 | 0,39 | 0,44 |

Des mesures supplémentaires aident. La politique dirigée (désaccord interne aux classes de primitives) ne fait pas mieux
que l'aléatoire en rappel ; le gain de précision sur les scènes multiples (0,44 contre 0,39) repose sur 8 scènes.
Résultat négatif pour la politique active.

## 7. Chaîne causale : primitives → candidats → récupération

### Matrice de décomposition oracle

| | Vraie composition | Composition WCE |
|---|---|---|
| **Vrais atomes** | identifiable dans 75 à 100 % des cas (borne) | candidat de la bonne forme : 0,75 à 1,00 ; candidat fonctionnel (≥ 0,97) : 0,00 à 1,00 ; récupération finale (≥ 0,99) : 0,00 à 0,25 |
| **Atomes découverts** (hybride) | accord 0,56 à 0,72 | candidat fonctionnel : 0,00 à 0,88 ; récupération finale : 0,00 à 0,13 |

Détail par profondeur, vrais atomes + composition WCE :

| | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Standard — candidat de la bonne forme | 1,00 | 0,88 | 0,88 | 0,75 |
| Standard — candidat fonctionnel | 0,88 | 0,50 | 0,38 | 0,00 |
| Standard — récupération finale | 0,13 | 0,00 | 0,00 | 0,00 |
| Standard — accord final moyen | 0,977 | 0,968 | 0,923 | 0,850 |
| Riche — candidat de la bonne forme | 1,00 | 0,88 | 1,00 | 0,88 |
| Riche — candidat fonctionnel | 0,88 | 1,00 | 0,75 | 0,75 |
| Riche — récupération finale | 0,25 | 0,13 | 0,00 | 0,00 |
| Riche — accord final moyen | 0,977 | 0,984 | 0,972 | 0,968 |

Atomes découverts + composition WCE (candidat fonctionnel) :

| | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Gradient, standard / riche | 0,25 / 0,50 | 0,25 / 0,38 | 0,00 / 0,25 | 0,00 / 0,00 |
| Hybride, standard / riche | 0,63 / 0,88 | 0,13 / 0,50 | 0,00 / 0,25 | 0,00 / 0,00 |
| Hybride minimal, standard / riche | 0,50 / 0,75 | 0,13 / 0,38 | 0,00 / 0,25 | 0,00 / 0,00 |
| Neuronal + hybride, standard / riche | 0,63 / 0,88 | 0,13 / 0,63 | 0,00 / 0,25 | 0,00 / 0,00 |

Rappel des primitives sur ces mêmes tâches (équivalence), standard d2 → d5 : gradient 0,81 → 0,25 ; hybride 0,56 → 0,15.
Riche : gradient 0,88 → 0,48 ; hybride 0,75 → 0,50.

### Transitions et corrélations (64 tâches par méthode)

| Méthode | Tâches où toutes les primitives sont retrouvées | Candidat fonctionnel si toutes retrouvées | … sinon | Récupération finale si candidat | … sinon | Corrélation rappel ↔ candidat | Corrélation rappel ↔ accord final |
|---|---|---|---|---|---|---|---|
| Gradient | 16 | 0,56 | 0,08 | 0,15 | 0,00 | 0,57 | 0,47 |
| Hybride | 7 | 0,86 | 0,23 | 0,05 | 0,00 | 0,57 | 0,50 |
| Hybride minimal | 7 | 0,86 | 0,18 | 0,25 | 0,00 | 0,54 | 0,39 |
| Neuronal + hybride | 8 | 1,00 | 0,21 | 0,05 | 0,00 | 0,64 | 0,51 |

Le manque d'atomes explique bien l'échec de génération : retrouver toutes les primitives multiplie par 4 à 7 la
probabilité d'avoir un candidat fonctionnel. Mais la transition suivante est mauvaise : avoir un candidat fonctionnel ne
donne la récupération finale que dans 5 à 25 % des cas. Sans candidat, jamais.

À la profondeur 2, les proposeurs hybrides donnent plus souvent un candidat fonctionnel que le gradient (0,63–0,88 contre
0,25–0,50) alors que leur rappel de primitives est plus bas : la composition tolère des primitives approximatives. C'est
le seul point où les nouveaux proposeurs font mieux en aval, et il ne se traduit pas en récupération finale.

## 8. Familles, bibliothèque, naissance, fusion, retrait

À partir des primitives validées du banc atomique (hybride, contraste ≥ 5) :

- **Familles paramétriques** : 8 familles (une par type), par exemple `norm.le(θ0∈[−4,07 ; 2,39], θ1∈[−3,05 ; 3,09],
  θ2∈[0,07 ; 3,52]) ×127`. Les plages couvrent presque tout le domaine : ces familles n'apportent aucune information
  au-delà du type. Le passage instances → famille → paramétrisation compacte est mécaniquement réalisé, pas utile.
- **Naissances proposées** : 5 relations récurrentes détectées, 2 acceptées — deux disques concentriques (3 tâches,
  2 paramètres économisés) et deux demi-plans parallèles (5 tâches, 1 paramètre). Ce sont l'anneau et la bande : le
  mécanisme retrouve des atomes que la grammaire possède déjà. Trois propositions vues sur une seule tâche sont
  refusées (pas de naissance opportuniste). Aucune naissance n'a été validée par un gain de prédiction ou de
  compression mesuré en aval.
- **Retrait** : aucun type candidat au retrait (tous sont validés au moins deux fois).
- **Fusion** : réalisée à la découverte par les classes d'équivalence.
- **Bibliothèque persistante** : fréquences d'usage des types (disque 39 %, demi-plan 16 %, secteur 13 %…). Son effet sur
  le temps de découverte, le rappel ou le transfert vers de nouveaux mondes n'a **pas été mesuré** : aucune seconde
  passe utilisant la bibliothèque n'a été lancée.

## 9. Erreurs et résultats négatifs

- Les nouveaux proposeurs ne battent pas le gradient direct sur le banc atomique.
- Le proposeur neuronal n'apporte aucun rappel et ne généralise pas hors distribution.
- L'objectif de contraste seul est la plus mauvaise méthode (équivalence 0,34, précision 0,12).
- Le validateur laisse passer beaucoup de fausses primitives (précision 0,12 à 0,37 hors réseau). Mécanismes
  identifiés : une frontière tangente ou parallèle à une vraie frontière sépare de vraies paires opposées ; des atomes
  à deux frontières (bandes, anneaux) balaient plusieurs frontières à la fois.
- Premier validateur sans correction de comparaisons multiples : trois primitives retenues sur des étiquettes
  aléatoires. Corrigé (seuil √(2·ln N) + 1) avant le run.
- Reconnaissance des macros (anneau, secteur) : quasi nulle.
- Politique active dirigée : pas mieux que l'aléatoire.
- Curriculum : pas d'effet positif mesuré.

## 10. Limites

- 8 tâches par cellule pour la chaîne aval et les décalages : résolution de 1/8.
- Le seuil de récupération finale (accord ≥ 0,99 sur [−3, 3]², zone d'extrapolation comprise) est sévère ; l'accord
  moyen est rapporté à côté.
- La perte entre « candidat fonctionnel » et « récupération finale » n'a pas été décomposée (polissage des paramètres,
  simplification, classement). Le pré-enregistrement interdisait de modifier ces étages pendant ce sous-jalon.
- Un seul réseau, une seule architecture, un seul budget d'entraînement.
- La base apprise, l'effet de la bibliothèque et la naissance validée en aval ne sont pas mesurés.
- Un seul monde synthétique en deux dimensions.

## 11. Prochaines étapes

1. Décomposer la perte aval avec les vrais atomes : candidat de la bonne forme dans 75 à 100 % des cas, récupération
   finale 0 à 25 %. C'est mesurable sans toucher au lot scellé et c'est maintenant le goulot le mieux isolé.
2. Découverte : combiner le gradient direct (meilleur sur les scènes simples) et le géométrique (meilleur sur les scènes
   à plusieurs atomes), et durcir le validateur contre les frontières tangentes.
3. Travailler en base minimale : même rappel, meilleure précision, 2,4 fois moins de temps.
4. Proposeur neuronal : le garder comme filtre de haute précision, ou l'abandonner ; rien ne justifie de l'agrandir.

## Readiness for WCE.01

Évaluation objective, composant par composant.

| Composant | État | Preuve |
|---|---|---|
| Primitive discovery | **manque — bloquant** | meilleure équivalence 0,56 ; précision 0,12 à 0,37 hors réseau ; anneaux et secteurs non retrouvés ; rappel 0,15 à 0,25 dans des compositions de profondeur 5 en régime standard |
| Compositional reconstruction | **partiel — bloquant** | avec les vrais atomes, la bonne forme est candidate dans 75 à 100 % des cas, mais la récupération finale ne dépasse pas 25 % |
| Fragment learning | **existe, non utile** | la bibliothèque couvre 58 à 67 % des sous-structures mais le gain de compression net est négatif ; le monde n'a pas de régularité de composition |
| Uncertainty | **partiel** | classes d'équivalence, hypothèses multiples de primitives et de structures, quatre niveaux d'incertitude implémentés ; aucun n'est calibré ni évalué en confirmatoire |
| Model-class boundary | **implémenté, non évalué** | onze signaux hors classe et décision à trois issues existent ; la frontière n'a été ni calibrée sur validation ni testée |
| Compact neural proposal | **existe, sans gain** | 23 881 paramètres, précision 0,54, rappel 0,22, pas de généralisation hors distribution |
| Lifelong library growth | **manque** | familles et naissances proposées ; aucun effet mesuré sur une seconde passe, aucune croissance validée |

**Prêt** : la représentation hiérarchique, la synthèse de composition sur signatures, le moteur d'ajustement par lots,
l'instrumentation de reachability, l'infrastructure AWS (checkpoints, reprise, progression), la discipline de
pré-enregistrement et de lot scellé.

**Manque** : une découverte de primitives fiable ; un validateur précis ; la reconnaissance des atomes à deux
frontières ; une récupération finale fidèle une fois les atomes connus ; l'évaluation de la frontière de classe ; une
bibliothèque dont la croissance améliore quelque chose.

**Bloquant** : (1) la découverte de primitives ; (2) la perte entre candidat correct et récupération finale. Tant que
ces deux points ne sont pas résolus, la thèse « primitives + transformations + composition + invariants → structure
complexe » n'est démontrée que pour l'étage de composition isolé, avec des atomes fournis.

**Verdict : WCE.01 n'est pas prêt.** Le lot scellé reste intact et la campagne confirmatoire 16 graines n'a pas été
lancée ; les lancer dans cet état reviendrait à consommer le lot scellé pour confirmer un échec déjà mesuré en dev.

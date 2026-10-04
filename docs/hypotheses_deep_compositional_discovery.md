# Pré-enregistrement — WCE Deep Compositional Structure Discovery & Model-Class Boundary

Rédigé le 2026-10-04, AVANT toute expérience quantitative de ce jalon. Branche `jalon-deep-compositional-discovery`,
partie du tag gelé `jalon-robust-reconstruction`. Le lot scellé du jalon précédent est consommé : il n'est utilisé pour
aucun réglage, diagnostic ou choix d'hyperparamètre.

Toute analyse ajoutée après observation de résultats est marquée `POST-HOC / EXPLORATORY` et reste séparée des
analyses confirmatoires ci-dessous.

## 0. Audit du code existant (lecture du code, AVANT expérience — à vérifier par instrumentation)

Causes candidates de l'effondrement au-delà de la profondeur 1 dans le jalon précédent. Ce sont des hypothèses de
lecture de code ; la section 16 de la spécification (profilage par profondeur) doit les départager.

1. Génération : la recherche est une chaîne gauche gloutonne (3 parents × 3 combinateurs × 8 atomes). Les arbres
   équilibrés ne sont jamais générés ; une sous-structure découverte n'est pas réutilisée.
2. Déclenchement : l'approfondissement n'a lieu que si le résidu dépasse le bruit attendu (test binomial) ; avec 8–12 %
   de bruit connu, il peut ne jamais se déclencher au-delà de la profondeur 2.
3. Ajustement : un nouvel atome reçoit une seule initialisation aléatoire ; l'ajustement joint par gradient reste
   probablement dans un optimum local.
4. Classement : pénalité de type BIC sur 120 points, qui favorise les structures courtes.
5. Élagage : l'ensemble de structures est tronqué après chaque niveau.
6. Identifiabilité : les observations passives ne couvraient que x ≤ 0,5 alors que la récupération était jugée sur
   [−3, 3]² ; 16 requêtes ne suffisent pas à identifier une structure profonde hors de la zone observée.
7. Le bruit du monde était FOURNI au système (taux vrai) ; il n'était pas estimé.

## 1. Jeu d'évaluation (`core/deep_world.py`)

- Profondeur compositionnelle d = nombre d'atomes ∈ {1, …, 6} ; chaque atome est effectif (son retrait modifie ≥ 4 %
  de la région observée). Arbres de formes variées (chaînes et arbres équilibrés), combinateurs and / or / xor.
- Régimes : `rich` (600 observations, bruit 2 %), `standard` (300, 5 %), `hard` (160, 10 %). Bruits : i.i.d., concentré
  aux frontières, outliers groupés. Le bruit n'est PAS fourni au nouveau système (il l'est à la baseline gelée, ce qui
  l'avantage).
- Observations passives sur x ≤ 1,2 ; extrapolation jugée sur x > 1,2 ; mesures actives permises partout.
- Partitions à graines disjointes : dev (base 1 000 000), validation (2 000 000), lot scellé (3 000 000), test
  confirmatoire (4 000 000).
- Compositions non vues : une signature globale sur quatre (d ≥ 3) est réservée ; elle n'apparaît jamais en dev ni en
  validation. Les sous-structures à 2 atomes restent vues.
- Hors classe (6 catégories) : primitive absente, composition hors budget, relation absente, dynamique incompatible,
  décalage de distribution, partiellement représentable. Difficiles mais valides (3) : bruit extrême (25 %), profond
  (5–6), peu de données (80).
- Lot scellé : 72 tâches (36 dans la classe, 12 difficiles-valides, 24 hors classe). Empreinte gelée avant tout
  réglage : `d57e9d761d165ef9d86ec7ee5c9c5c7909c4b1622425c2d2379cbdddaf1a6ca2`. Évaluation unique en fin de jalon.

## 2. Méthodes comparées

- `baseline` : système du jalon précédent, GELÉ (`RobustStructuralReconstruction`, politique EFA… inchangés).
- `bottom_up` : sous-structures simples ajustées puis composées (y compris entre elles), initialisation par le résidu.
- `top_down` : décomposition récursive de la cible en (atome, combinateur, cible résiduelle avec support restreint).
- `bidirectional` : cibles résiduelles (descendantes) appariées à la bibliothèque de sous-structures (montantes).
- `best_first` : recherche ordonnée par un score explicite (évidence, MDL, nouveauté fonctionnelle, réutilisation,
  coût de recherche).
- Méthode PRIMAIRE : une seule stratégie, choisie sur la VALIDATION (récupération fonctionnelle moyenne d = 2–4, régime
  standard), figée avant le test confirmatoire. Si une variante s'avère redondante ou infaisable en dev, elle est
  abandonnée et la décision documentée.
- Ablations de la méthode primaire : sans bibliothèque de fragments, sans anti-unification, sans classes d'équivalence,
  sans NONE_OF_THE_ABOVE, sans hiérarchie (candidats plats, pas de réutilisation), sans requête d'identifiabilité.
- Politiques de mesure : réduction de classe d'équivalence (nouvelle), IG structurel, désaccord, aléatoire, EFA
  historique, oracle.

### Budget de calcul (architecture contre calcul)

L'unité est l'« équivalent-ajustement » : une initialisation × 100 pas de gradient d'un candidat. Deux budgets :

- `apparié` : plafond égal à la consommation moyenne de la baseline sur les mêmes tâches (mesurée en dev, figée
  ensuite). C'est le budget des comparaisons CONFIRMATOIRES.
- `étendu` : 3 × le budget apparié. Rapporté à part ; un gain qui n'apparaît qu'au budget étendu est attribué au
  calcul, pas à l'architecture.

Le temps mur et la mémoire sont rapportés en plus, par profondeur.

## 3. Métriques (jamais fusionnées en une seule)

- `ExactStructure` : squelette canonique identique au vrai.
- `EquivalentStructure` : identique à réécritures algébriques près (commutativité, associativité, complément).
- `FunctionallyEquivalent` : accord ≥ 0,99 sur 4 000 sondes de [−3, 3]².
- `PredictivelyAdequate` : accord ≥ 0,95 en région observée ET en extrapolation.
- Extrapolation : accord sur x > 1,2. Invariants conservés : part des invariants du programme vrai retrouvés.
- Par profondeur : toutes les métriques sont rapportées séparément pour d = 1, 2, 3, 4, 5+.
- Décision à trois issues : `KNOWN`, `UNKNOWN` (information insuffisante), `NONE_OF_THE_ABOVE` (observations
  incompatibles avec les hypothèses disponibles).

Statistiques : 16 graines de test, moyenne par graine, IC 95 % (t de Student), différences appariées par graine.

## 4. Hypothèses confirmatoires

« Soutenue » exige le seuil ET un IC 95 % excluant 0 dans le bon sens, sauf mention contraire. Entre « soutenue » et
« réfutée » : non concluante.

| # | Hypothèse | Mesure | Comparaison | Soutenue si | Réfutée si |
|---|---|---|---|---|---|
| H1 | Récupération profonde | `FunctionallyEquivalent`, moyenne d = 2–4, régime standard, budget apparié | primaire − baseline | ≥ +0,15 | ≤ +0,03 |
| H2 | Chute avec la profondeur | rapport récupération(d = 2–4) / récupération(d = 1) | primaire − baseline | ≥ +0,20 | ≤ 0 |
| H3 | Efficacité de recherche | part des tâches d = 2–4 où le vrai squelette (à équivalence près) est GÉNÉRÉ, budget apparié | primaire − baseline | ≥ +0,20 | ≤ 0 |
| H4 | Fausse erreur structurelle | parmi les erreurs `ExactStructure` de la primaire (d ≥ 2), part où la structure retenue est dans la même classe d'équivalence observationnelle que la vraie | seuil absolu | ≥ 0,30 | < 0,10 |
| H5 | Requête d'identifiabilité | taille finale moyenne de la classe d'équivalence, 16 requêtes | réduction de classe contre IG structurel | réduite ≥ 25 % sans perte fonctionnelle > 0,02 | réduction ≤ 0 |
| H6 | Détection hors classe | fausse acceptation (hors classe déclaré `KNOWN`) et faux rejet (valide déclaré `NONE_OF_THE_ABOVE`) | seuils absolus ; baseline rapportée | FA ≤ 0,30 ET FR ≤ 0,15 | FA > 0,50 OU FR > 0,30 |
| H7 | Calibration de l'abstention | ECE de P(hors classe) sur le test, calibré sur validation ; pureté : part de `NONE_OF_THE_ABOVE` parmi les non-`KNOWN` hors classe, part de `UNKNOWN` parmi les non-`KNOWN` valides | seuils absolus | ECE ≤ 0,10 ET les deux puretés ≥ 0,70 | ECE > 0,20 OU une pureté < 0,50 |
| H8 | Complexité | excès de taille absolu moyen (atomes retenus − vrais), d = 1–4 ; violations du budget compact (64 paramètres) | primaire contre baseline | excès ≤ baseline ET 0 violation | excès > baseline + 0,5 OU ≥ 1 violation |
| H9 | Extrapolation | accord sur x > 1,2, d = 1–4 | primaire − baseline | ≥ −0,02 (non-infériorité : borne basse de l'IC ≥ −0,02) | moyenne < −0,05 |
| H10 | Coût de calcul | (a) accélération de l'ajustement par équivalent-ajustement ; (b) écart numérique maximal avec l'ajustement de référence | nouveau moteur contre `fit_robust` gelé | (a) ≥ 3× ET (b) ≤ 1e-6 sur les paramètres finaux | (a) < 1,5× OU (b) > 1e-3 |
| H11 | Réutilisation de fragments | équivalents-ajustement jusqu'à la première structure adéquate, banc « compositions non vues » | avec bibliothèque (apprise en dev) contre sans | réduits ≥ 20 % sans perte de récupération > 0,02 | pas de réduction OU perte > 0,05 |
| H12 | Compositions non vues | `FunctionallyEquivalent`, d = 3–4, régime standard | non vues − vues (primaire) | écart ≥ −0,10 (borne basse de l'IC ≥ −0,10) | moyenne < −0,20 |
| H13 | Anti-unification | banc synthétique dédié : généralisation correcte et surgénéralisation | seuils absolus | correcte ≥ 0,80 ET surgénéralisation ≤ 0,10 | correcte < 0,60 OU surgénéralisation > 0,25 |

Garde-fou de compacité pour H1, H3 et H11 : un gain obtenu avec plus de 2× les candidats générés par la baseline n'est
pas compté comme victoire architecturale (il est rapporté, avec cette mention).

## 5. Analyses confirmatoires secondaires (rapportées sans seuil)

- Récupération par profondeur (1, 2, 3, 4, 5+) et par régime, toutes méthodes, budgets apparié et étendu.
- Devenir de la vraie structure, par profondeur : jamais générée / générée mais mal ajustée / bien ajustée mais mal
  classée / équivalente à d'autres / supprimée par élagage — pour la baseline et la primaire.
- Par profondeur : candidats générés, candidats ajustés, familles, classes d'équivalence, taux d'élagage, temps de
  recherche, temps d'ajustement, mémoire, rang de la vraie structure.
- Signaux hors classe pris SÉPARÉMENT (AUROC, AUPRC), puis combinaison transparente (régression logistique ajustée sur
  validation uniquement).
- Par catégorie hors classe et par catégorie difficile-valide : matrice KNOWN / UNKNOWN / NONE_OF_THE_ABOVE.
- Ablations de la primaire.
- Taille moyenne des classes d'équivalence par profondeur, avant et après mesures.

## 6. Analyses exploratoires (annoncées, non confirmatoires)

- Diagnostic de primitive manquante sur les mondes hors classe.
- Profondeur 6 (test de résistance).
- Bibliothèque de fragments : fréquence, gain de compression, gain de recherche, transfert.
- Régimes où d = 3–5 devient identifiable.

## 7. Règles de réglage et de fuite

- Tout réglage se fait en dev ; sélection de la primaire et calibration sur validation ; test confirmatoire lancé une
  fois la configuration figée (empreinte de configuration enregistrée dans le manifeste).
- Le lot scellé n'est ni lu, ni tracé, ni résumé avant l'évaluation finale unique.
- Le code d'inférence n'importe jamais `core/deep_world.py` ; les diagnostics qui utilisent la vérité sont dans un
  module d'évaluation séparé et ne renvoient rien au système.
- La bibliothèque de fragments n'est apprise que sur dev.

## 8. Critères d'arrêt et de réduction (décidés d'avance)

- Avant toute campagne 16 graines : micro-benchmark du nouveau système, coût par profondeur connu, génération de
  candidats instrumentée.
- Si le coût estimé de la campagne confirmatoire dépasse 10 h sur 8 vCPU, réduire dans cet ordre : (1) profondeur 6
  limitée au régime standard ; (2) ablations sur 8 graines ; (3) régime `rich` sur 8 graines. Les hypothèses H1–H3
  gardent toujours 16 graines.
- Une stratégie dont le coût par tâche dépasse 5× celui de la baseline en dev est abandonnée (décision documentée).
- Si la primaire ne dépasse pas la baseline en validation, le test confirmatoire est QUAND MÊME exécuté et rapporté.

## 9. Conditions d'échec du jalon (explicites)

Le jalon échoue sur son critère architectural si H1 et H2 sont réfutées, ou si le gain n'apparaît qu'au budget étendu
ou avec plus de 2× les candidats de la baseline. Dans ce cas le rapport doit dire où la vraie structure se perd
(section 5) et pourquoi.

## 10. Addendum de conception (2026-10-04, AVANT toute expérience quantitative)

Précisions de protocole fixées avant le premier run, pour qu'elles ne soient pas choisies après coup.

- Coût décisionnel (H7, « coût d'abstention ») : décision correcte 0 ; `UNKNOWN` 1 ; `NONE_OF_THE_ABOVE` sur un monde
  valide 3 ; `KNOWN` sur un monde hors classe 5 ; `KNOWN` avec une structure fonctionnellement fausse sur un monde
  valide 2. Les seuils de décision sont choisis sur validation pour minimiser ce coût, puis figés.
- Décision correcte : hors classe → `NONE_OF_THE_ABOVE` ; valide → `KNOWN` avec une structure `FunctionallyEquivalent`,
  ou `UNKNOWN` si la structure retenue ne l'est pas.
- Budgets de mesure du nouveau système, par tâche : 16 requêtes d'identification (capteur standard) ; 24 mesures de
  validation de frontière au capteur précis (12 en zone observée, 12 hors zone) ; 24 répétitions de points passifs
  (estimation du bruit sans modèle). La baseline garde son budget du jalon précédent (16 unités) et reçoit le bruit
  vrai. Les mesures de validation ne servent PAS à l'ajustement des structures.
- Retenue interne : 20 % des observations passives sont retirées avant la recherche et servent au contrôle prédictif.
- Classe d'équivalence observationnelle : structures ajustées dont les prédictions s'accordent sur ≥ 99 % des points
  mesurés (liaison simple), dans une fenêtre d'évidence de log 20 autour de la meilleure.
- Le bruit n'est pas fourni au nouveau système : chaque hypothèse profile son propre taux de retournement.
- Signaux hors classe évalués séparément : contrôle prédictif sur retenue, regroupement spatial du résidu, écart à un
  modèle non paramétrique (k plus proches voisins), longueur de description par observation, instabilité des
  paramètres, amélioration encore en cours à la profondeur maximale, dérive temporelle du résidu, validation au
  capteur précis en zone et hors zone, excès de bruit par rapport aux répétitions, désaccord structurel persistant.

## 11. Gel (2026-10-04)

Ce pré-enregistrement et le manifeste du lot scellé (`docs/sealed_deep_manifest.json`, empreinte
`d57e9d761d165ef9d86ec7ee5c9c5c7909c4b1622425c2d2379cbdddaf1a6ca2`) sont commités avant la poursuite du jalon. À partir de ce commit, le lot scellé est INTERDIT à toute
inspection adaptative jusqu'à l'évaluation finale unique. Au moment du gel, seuls ont tourné : des essais de fumée
locaux sur quelques tâches de dev et un micro-benchmark AWS de dev (2 tâches par profondeur) ; aucun n'a touché la
validation, le test ni le lot scellé.

# Rapport diagnostique — WCE Deep Candidate Generation

Sous-jalon EXPLORATOIRE du jalon « Deep Compositional Structure Discovery & Model-Class Boundary ».
Données : partition **dev uniquement**. Ni la validation, ni le test confirmatoire, ni le lot scellé n'ont été touchés.
Rien ici n'est confirmatoire : toutes les analyses sont `POST-HOC / EXPLORATORY` au sens du pré-enregistrement
(`docs/hypotheses_deep_compositional_discovery.md`, gelé au tag `prereg-deep-compositional-discovery`).

Sources : `results/deep_generation_summary.json` (run AWS `wce-deepgen-20261004-134629`, 1 056 sous-unités, 8 tâches par
profondeur et par régime), `results/deep_cover_summary.json` (run `wce-deepcover-20261004-142805`, 6 tâches par
cellule), `results/deep_reachability.json`, `results/deep_coverage.json`, `results/deep_fragments.json`.
Cibles : `make deep-reachability`, `deep-coverage`, `deep-fragments`, `deep-run-gen`, `deep-run-cover`,
`deep-gen-aggregate`, `deep-cover-aggregate`, `deep-gen-report`.

Effectifs : 8 tâches par cellule (profondeur × régime) pour les rappels, 6 pour la couverture exhaustive. Les
proportions ci-dessous ont donc une résolution de 1/8 ou 1/6 ; elles situent le problème, elles ne le chiffrent pas
finement.

## Résumé

La vraie structure n'entre pas dans les candidats au-delà de la profondeur 2. Le diagnostic sépare cinq étages :

| Étage | Verdict | Mesure |
|---|---|---|
| Représentation | n'est pas le problème | 100 % des structures vraies représentables, intermédiaires tous valides |
| Composition | résolue QUAND les atomes sont connus | avec les vrais atomes, la vraie forme est générée dans 100 % des cas jusqu'à la profondeur 3, 75 à 88 % aux profondeurs 4–5, 50 % à la profondeur 6 |
| Identifiabilité | suffisante en régimes standard et riche | la vraie structure (vrais paramètres) bat ses variantes privées d'un atome dans 75 à 100 % des cas |
| Classement | n'est pas le problème observé | aucune cause `RANKING_FAILURE` sur 640 runs de profondeur 3–6 |
| Ajustement / découverte des atomes | **goulot** | rappel des vrais atomes 0,03 à 0,12 au-delà de la profondeur 1 ; même en ajustant TOUTES les formes, la vraie forme n'est jamais sélectionnée à la profondeur 3 |

Le problème n'est donc ni la grammaire, ni la logique de composition, ni le classement : c'est la localisation des
paramètres des atomes sous composition. Le sous-jalon suivant est la découverte de primitives.

## 1. Pourquoi la vraie structure n'était-elle pas générée à profondeur 3–6 ?

Trois causes mécaniques, mesurées séparément.

**(a) Les opérateurs.** La baseline, la stratégie plate et la descendante n'ajoutent qu'un atome à la fois : elles ne
construisent que des chaînes. Part des structures vraies atteignables par ces opérateurs (120 structures par
profondeur, analyse sans ajustement) : 100 % jusqu'à la profondeur 3, 90 % à 4, 77 % à 5, 63 % à 6. L'opérateur
manquant est la composition de deux sous-structures (`compose_pair`).

**(b) La taille de l'espace.** Nombre de formes distinctes : 8, 108, 2 088 (énumération exacte), puis ≈ 8,0·10⁴,
2,8·10⁶, 1,0·10⁸ (borne). Les stratégies génèrent 110 à 220 candidats. Couverture exacte mesurée de l'espace à la
profondeur 2 : 58 % (baseline, plate), 16 % (montante), 11 % (bidirectionnelle), 3 % (factorisée) ; à la profondeur 3 :
1,7 à 2,9 % (baseline, plate), moins de 1 % pour les autres.

**(c) Les atomes.** Les stratégies guidées par le résidu ne cherchent pas au hasard : elles suivent les atomes qu'elles
ont ajustés. Or ces atomes ne sont pas les vrais (section 3). Cause principale relevée sur les runs de profondeur 3–6
(32 par méthode et par régime) :

| Méthode (budget 150) | `TRUE_PARENT_NOT_GENERATED` | `PRUNED_TOO_EARLY` | `BUDGET_EXHAUSTED` | `OPERATORS_INSUFFICIENT` | `FIT_FAILURE` | Récupérée |
|---|---|---|---|---|---|---|
| Baseline, standard | 20 | 7 | 0 | 4 | 0 | 1 |
| Baseline, riche | 24 | 3 | 0 | 4 | 0 | 1 |
| Plate, standard | 20 | 2 | 5 | 4 | 1 | 0 |
| Plate, riche | 24 | 0 | 2 | 4 | 0 | 2 |
| Montante, standard | 29 | 1 | 1 | 0 | 1 | 0 |
| Montante, riche | 31 | 0 | 1 | 0 | 0 | 0 |
| Bidirectionnelle, standard | 30 | 2 | 0 | 0 | 0 | 0 |
| Factorisée, standard | 32 | 0 | 0 | 0 | 0 | 0 |
| Factorisée, riche | 30 | 1 | 0 | 0 | 1 | 0 |

Aucune occurrence de `GRAMMAR_UNREPRESENTABLE`, `CANONICALIZATION_COLLAPSE`, `TYPE_MISMATCH`, `RANKING_FAILURE` ni
`OBSERVATIONAL_NON_IDENTIFIABILITY` à ces profondeurs. `FRAGMENT_MISSING` ne peut pas apparaître : aucune stratégie
n'utilisait de bibliothèque dans ce run.

## 2. Était-elle théoriquement atteignable ?

Oui pour la représentation : 100 % des structures vraies sont exprimables, et tous leurs préfixes, suffixes, sous-arbres
et compositions partielles sont des objets de recherche valides, sans collision de clé (clôture des parents : 100 %).

Oui pour les opérateurs dès qu'on dispose de `compose_pair` ou de la synthèse sur signatures : 100 % à toutes les
profondeurs. Non pour 10 à 37 % des structures de profondeur 4–6 avec les seuls opérateurs « un atome à la fois ».

Chemin oracle minimal : 2d − 1 opérations (1, 3, 5, 7, 9, 11). Largeur nécessaire (nombre de résultats partiels à
détenir) : 2 jusqu'à la profondeur 3, jusqu'à 3 au-delà.

Statut `Reachability` : `REPRESENTABLE_BUT_NOT_REACHED` pour la quasi-totalité des échecs ;
`UNREACHABLE_WITH_CURRENT_OPERATORS` pour 4 cas sur 32 (baseline et plate) ; `NOT_REPRESENTABLE` jamais.

## 3. Quel opérateur ou parent manquait ?

L'opérateur `compose_pair` pour les arbres équilibrés (section 1a). Mais le parent manquant le plus fréquent est de
taille 1 ou 2 : les ATOMES eux-mêmes. Rappel des vrais atomes (même type, accord ≥ 0,95 sur la région observée) parmi
48 atomes ajustés par gradient sur la cible :

| Profondeur | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Standard | 0,88 | 0,06 | 0,12 | 0,03 | 0,05 | 0,06 |
| Riche | 1,00 | 0,06 | 0,12 | 0,03 | 0,03 | 0,04 |
| Standard, seuil 0,90 | 1,00 | 0,38 | 0,21 | 0,12 | 0,10 | 0,10 |

L'objectif de séparation des paires de frontière essayé dans ce sous-jalon fait pire (0,00 à 0,08 au-delà de la
profondeur 1) : résultat négatif.

## 4. Le pruning supprimait-il le bon chemin ?

Marginalement. `PRUNED_TOO_EARLY` représente 0 à 7 cas sur 32 selon la méthode (surtout la baseline en régime
standard). Dans la grande majorité des cas, le bon chemin n'est pas élagué : il n'est jamais ouvert.

## 5. Le budget était-il suffisant ?

Pour une recherche non guidée, non, structurellement : 150 équivalents-ajustement contre 2 088 formes à la profondeur 3.
Mais augmenter le budget n'aide pas les stratégies actuelles : à 450, la montante consomme en moyenne 125 (standard) à
173 (riche) équivalents-ajustement et s'arrête sur sa règle d'absence de gain ; la factorisée consomme 129 à 136 quel que
soit le plafond. Les rappels à 150 et à 450 sont identiques. La courbe récupération / budget est plate.

L'énumération exhaustive montre que le budget n'est pas la vraie limite. En ajustant TOUTES les formes (4
initialisations chacune) :

| | Formes ajustées | Équivalents-ajustement | CPU | Vraie forme sélectionnée | Rang médian de la vraie forme | Accord de la meilleure forme |
|---|---|---|---|---|---|---|
| Profondeur 2, standard | 116 | 464 | 18 s | 17 % | 36 | 0,926 |
| Profondeur 2, riche | 116 | 464 | 19 s | 33 % | 40 | 0,979 |
| Profondeur 3, standard | 2 204 | 8 816 | 420 s | 0 % | 915 | 0,866 |
| Profondeur 3, riche | 2 204 | 8 816 | 470 s | 0 % | 821 | 0,932 |

Même avec 59 fois le budget de référence, la vraie forme de profondeur 3 n'est jamais sélectionnée : elle est générée
mais mal ajustée (son rang médian est autour de 900 sur 2 204). L'ajustement des paramètres depuis des initialisations
heuristiques échoue. C'est le même goulot que la section 3, vu de l'autre côté.

## 6. Les fragments aident-ils ?

Peu, et pas par construction. Borne haute (bibliothèque de 32 fragments construite sur les VRAIES structures de 120
tâches de curriculum de profondeur 2–3) : 58 à 67 % des sous-structures vraies à 2 atomes sont dans la bibliothèque ; le
chemin oracle ne raccourcit que de 8 à 12 % ; le gain de compression net est négatif (−166 bits sur 60 structures de
profondeur 4, la bibliothèque coûtant plus qu'elle n'économise). Le monde d'évaluation tire ses compositions au hasard :
il n'y a pas de régularité de composition à apprendre. Les fragments ne changent pas non plus la reachability une fois la
synthèse sur signatures disponible, puisqu'elle énumère déjà toutes les paires.

Non mesuré : l'effet des fragments sur les stratégies par résidu dans un run complet (les variantes « fragments exacts /
anti-unifiés / transférés » n'ont pas été lancées, le diagnostic ayant montré que le goulot est en amont).

## 7. L'anti-unification aide-t-elle ?

Comme compresseur de groupe, non : anti-unifier tous les fragments de même taille donne le schéma trivial (résultat
négatif, corrigé en anti-unification par paires). Comme générateur, modestement : les schémas à une variable issus des
paires de fragments proposent 30 schémas dont 16 absents de la bibliothèque, et la couverture des sous-structures vraies
passe de 58–67 % à 68–79 %. Coût : ces schémas couvrent une large part des 108 formes à 2 atomes ; c'est un élargissement
de la génération, pas une généralisation ciblée.

## 8. Le typage réduit-il réellement l'espace ?

Au niveau des atomes, oui : 30 nœuds unaires bien typés sur 45 bruts (un tiers écarté ; 70 % à la profondeur 3 si l'on
composait des nœuds bruts). Au niveau logique, non : tout couple de régions est composable par and / or / xor, le typage
n'écarte aucune composition. La réduction utile vient des signatures : sur un exemple de profondeur 3 avec 15 littéraux,
28 % des 2 727 compositions tentées sont fusionnées (même signature) et 30 % écartées comme dégénérées (constantes, ou
identiques à un enfant). Ce taux n'a pas été agrégé sur l'ensemble du run.

## 9. La bidirectionnelle rencontre-t-elle enfin le bon sous-espace ?

La bidirectionnelle par ajustements, non : rappel à la génération 0,12 à la profondeur 2, 0 au-delà ; couverture
quasi nulle à la profondeur 3. La rencontre sur signatures (tables montantes + trous typés descendants + produit
bilinéaire), oui, À CONDITION d'avoir les atomes :

| Profondeur | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Vraie forme générée, vrais atomes seuls (standard) | 1,00 | 1,00 | 1,00 | 0,75 | 0,88 | 0,50 |
| Avec 12 atomes parasites (standard) | 1,00 | 1,00 | 1,00 | 0,75 | 0,62 | 0,12 |
| … et classée première avant ajustement | 1,00 | 1,00 | 0,62 | 0,50 | 0,25 | 0,00 |
| Temps de synthèse | 0,02 s | 0,03 s | 0,05 s | 0,06 s | 0,19 s | 2,4 s |

Les échecs aux profondeurs 4–6 avec les vrais atomes ne sont pas expliqués en détail (hypothèses : borne sur le nombre de
littéraux, fusion par signature d'une forme équivalente, faisceau des trous). À la profondeur 6 avec parasites, la
synthèse ne suffit pas.

## 10. Quel est le Recall@generated par profondeur ?

| Méthode (budget 150) | d1 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|---|
| Baseline, standard | 1,00 | 0,88 | 0,12 | 0,00 | 0,00 | 0,00 |
| Plate, standard | 1,00 | 0,75 | 0,12 | 0,00 | 0,00 | 0,00 |
| Montante, standard | 1,00 | 0,25 | 0,12 | 0,00 | 0,00 | 0,00 |
| Bidirectionnelle, standard | 1,00 | 0,12 | 0,00 | 0,00 | 0,00 | 0,00 |
| Factorisée, standard | 1,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| Baseline, riche | 1,00 | 0,88 | 0,12 | 0,00 | 0,00 | 0,00 |
| Plate, riche | 1,00 | 1,00 | 0,25 | 0,00 | 0,00 | 0,00 |
| Montante, riche | 1,00 | 0,38 | 0,00 | 0,00 | 0,00 | 0,00 |
| Factorisée, riche | 1,00 | 0,00 | 0,12 | 0,00 | 0,00 | 0,00 |

Aucune amélioration : les nouvelles stratégies génèrent MOINS souvent la vraie forme que la baseline à la profondeur 2.
Les autres rappels (régime standard, baseline / plate / montante) : après élagage 0,50 / 0,62 / 0,25 à la profondeur 2 ;
bien ajustée 0,25 / 0,12 / 0,12 ; dans les 5 premières 0,50 / 0,12 / 0,12 ; sélectionnée 0,25 / 0,12 / 0,12. À partir de la
profondeur 4, tous les rappels sont nuls pour toutes les méthodes. Le rappel avant élagage n'est pas mesuré séparément :
il est égal au rappel à la génération dans cette instrumentation.

L'objectif diagnostique exploratoire (« amélioration substantielle à la profondeur 3, fraction non triviale à la
profondeur 4 ») n'est PAS atteint par la recherche complète. Il n'est atteint que par l'étage logique isolé (section 9).

## 11. Coût par vraie structure rendue atteignable

Par la recherche complète : non défini (aucune structure de profondeur ≥ 4 rendue atteignable). Par l'étage logique avec
vrais atomes : 0,05 à 0,2 s de CPU et 10³ à 10⁶ paires évaluées par structure jusqu'à la profondeur 5, sans aucun
ajustement par gradient.

Coûts moyens séparés (régime standard, toutes profondeurs) :

| Méthode | Équivalents-ajustement | Candidats | Compositions tentées | CPU total | dont ajustement | dont synthèse |
|---|---|---|---|---|---|---|
| Baseline | 132 | 109 | 0 | 40,0 s | 40,0 s | 0 |
| Plate @150 | 127 | 119 | 0 | 17,0 s | 16,5 s | 0 |
| Montante @150 | 116 | 173 | 0 | 12,4 s | 12,1 s | 0 |
| Bidirectionnelle @150 | 89 | 141 | 0 | 8,1 s | 7,9 s | 0 |
| Factorisée @150 | 129 | 217 | 3,5·10⁸ | 20,5 s | 7,2 s | 13,2 s |

Pour la factorisée, la synthèse (génération) est devenue le poste dominant, devant l'ajustement : le point chaud n'est
plus `fit`. L'élagage, la canonicalisation, le regroupement d'équivalence et la recherche de fragments ne sont pas
chronométrés séparément (limite de l'instrumentation).

## 12. Le gain vient-il de meilleure composition ou de plus de calcul ?

Il n'y a pas de gain de récupération à attribuer. Ce qui est établi : (i) plus de calcul n'aide pas (budget 450 = budget
150 ; l'énumération exhaustive à 8 816 équivalents-ajustement ne sélectionne pas la vraie forme) ; (ii) la composition
par signatures résout l'étage logique en une fraction de seconde quand les atomes sont justes. Le facteur limitant n'est
ni la composition ni le calcul : ce sont les atomes.

## Identifiabilité, distinguée de la génération

Avec la vraie structure et ses vrais paramètres, sans recherche : part des tâches où elle bat toutes ses variantes privées
d'un atome (facteur de Bayes > 3).

| Régime | d2 | d3 | d4 | d5 | d6 | Source |
|---|---|---|---|---|---|---|
| Difficile (160 points, bruit total 15 %) | 0,70 | 0,47 | 0,15 | 0,07 | 0,03 | 40 tâches, local |
| Standard (300 points, 10 %) | 1,00 | 0,93 | 0,88 | 0,70 | 0,72 | 40 tâches, local |
| Standard | 1,00 | 1,00 | 0,88 | 0,88 | 0,75 | 8 tâches, AWS |
| Riche (600 points, 7 %) | 1,00 | 1,00 | 0,97 | 1,00 | 0,97 | 40 tâches, local |
| Riche | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 8 tâches, AWS |

En régime difficile, les profondeurs 4–6 ne sont pas identifiables quelle que soit la recherche : l'atome le plus faible
modifie une douzaine de points pour 24 étiquettes retournées attendues. En régimes standard et riche, l'échec n'est pas
un problème d'identifiabilité.

## Qualité de la base

La base est redondante : bande = deux demi-plans parallèles, anneau = disque ∧ anti-disque, secteur = deux demi-plans par
l'apex. Une structure vraie compte en moyenne 1,5 fois plus d'atomes en base minimale {demi-plan, disque} (1,67 ; 3,07 ;
4,52 ; 6,09 ; 7,58 ; 9,07 pour les profondeurs 1 à 6). Une structure peut donc être retrouvée fonctionnellement sous une
autre forme, comptée comme un échec de rappel de forme. Aucune structure n'exige une primitive absente ; aucune n'est
artificiellement longue par rapport à sa profondeur déclarée.

## Canonicalisation

Aucun cas de `CANONICALIZATION_COLLAPSE` mesuré. Deux précautions vérifiées par tests : la clé d'une sous-structure
n'applique aucune réécriture dépendant des paramètres (sinon un `translate(0, 0)` ajusté changerait de clé) ; deux formes
associativement équivalentes gardent des clés distinctes pendant la construction et ne sont réunies qu'à l'évaluation.
Dans la synthèse, des formes de même signature sont fusionnées mais jusqu'à deux alternatives sont conservées.

## Bugs trouvés

- Dans les lignes de l'exploration dev précédente, le booléen « vraie structure générée » était écrasé par le nombre de
  candidats générés ; le champ `fate` restait correct. Corrigé pour ce sous-jalon (champs distincts).
- La bibliothèque de fragments canonisait les programmes ajustés avec une réécriture dépendant des paramètres : un même
  fragment recevait deux clés. Corrigé (`normalize`).
- Premier essai du générateur factorisé sans borne sur le nombre de littéraux : tables de plusieurs millions de lignes,
  plus de 10 minutes sans résultat. Corrigé par une sélection de diversité (14 littéraux au plus).

## Limites

- 8 tâches par cellule (6 pour la couverture) : faible résolution statistique.
- Le rappel avant élagage n'est pas distinct du rappel à la génération ; les temps d'élagage, de canonicalisation et
  d'équivalence ne sont pas séparés.
- Les échecs de l'étage logique aux profondeurs 4–6 avec les vrais atomes ne sont pas expliqués en détail.
- La détection géométrique directe (droites et cercles par consensus) et l'objectif de séparation ont échoué ; ils ne
  sont décrits ici que comme résultats négatifs.
- Les variantes « fragments » des stratégies par résidu et le curriculum de profondeur n'ont pas été exécutés en
  conditions réelles.

## Réponse à la question de sortie

Pourquoi les vraies structures profondes n'étaient pas générées : parce que les paramètres des atomes ne sont pas
trouvés sous composition (rappel atomique ≈ 0,05), ce qui laisse la recherche guidée sans prise et rend l'énumération
exhaustive elle-même inopérante ; accessoirement parce que les opérateurs « un atome à la fois » excluent 10 à 37 % des
arbres de profondeur 4–6.

Quelles modifications minimales rendent la génération possible sans force brute : la synthèse sur signatures (tables
montantes, trous typés, rencontre bilinéaire) suffit pour la composition jusqu'à la profondeur 5 — MAIS seulement une
fois les atomes disponibles. Cette condition n'est pas remplie aujourd'hui. Le critère de sortie du sous-jalon n'est donc
atteint qu'à moitié : la cause est établie expérimentalement, le remède complet ne l'est pas. Le goulot suivant est la
découverte de primitives (`docs/rapport_primitive_discovery.md`).

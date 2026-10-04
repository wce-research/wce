# Pré-enregistrement — jalon « Boundary Program Induction » (figé avant toute exécution)

Date de gel : 2026-10-02. Critères NON modifiables après observation. Unité : graine (≥ 16). IC95 par t sur graines.
« Représentable » = exprimable par l'algèbre de primitives à la profondeur maximale de recherche.
Familles held-out = absentes de tout code d'induction (aucun nom, aucun helper dédié) ; présentes seulement dans le générateur du banc.

| H | Métrique | Baseline | Succès | Échec |
|---|---|---|---|---|
| H1 reconstruction d'une famille held-out représentable | récupération FONCTIONNELLE (accord ≥ 0,97 avec la vraie frontière sur la région d'évaluation complète) moyenne sur familles held-out représentables | known-family oracle fitter (borne haute) | ≥ 0,6 des instances récupérées ET ≥ 3 familles held-out distinctes récupérées au moins une fois | < 0,3 |
| H2 extrapolation | exactitude HORS région observée | petit MLP (capacité comparable) et mémorisation (1-NN) | BPI > MLP + 0,05 ET BPI > 1-NN + 0,05, IC95 de la différence appariée > 0 | BPI ≤ MLP (IC contient ou sous 0) |
| H3 famille paramétrée réutilisable | sur nouvelles instances d'une famille induite par anti-unification : exactitude via la macro vs recherche complète ; coût (évaluations) | recherche complète sans anti-unification | exactitude macro ≥ recherche − 0,02 ET évaluations ≤ 0,5 × recherche | exactitude macro < recherche − 0,05 |
| H4 expériences actives | nb d'observations pour identifier le programme (P ≥ 0,95 et correct) | requêtes aléatoires | médiane(actif) ≤ 0,6 × médiane(aléatoire) ET taux d'identification actif ≥ aléatoire | médiane(actif) ≥ médiane(aléatoire) |
| H5 TruthAnchors | taux de réfutation des certificats auto-cohérents FAUX ; fausses réfutations des certificats corrects | certificat sans ancre (0 réfutation par construction) | réfutation ≥ 0,7 ET fausses réfutations ≤ 0,1 | réfutation < 0,3 |
| H6 cristallisation | fraction d'experts neuronaux remplacés par un programme ; perte d'exactitude moyenne des remplacés | experts conservés | fraction ≥ 0,4 sur familles représentables ET perte ≤ 0,02 | fraction < 0,15 ou perte > 0,05 |
| H7 programme + résidu | exactitude et paramètres sur fonctions partiellement symbolisables | expert neuronal seul | exactitude(prog+résidu) ≥ expert − 0,01 ET paramètres ≤ 0,5 × expert | exactitude < expert − 0,05 |
| H8 non-représentable reconnu | taux d'abstention « aucun programme adéquat » sur fonctions non représentables ; faux rejets sur représentables | — | abstention ≥ 0,8 ET faux rejet ≤ 0,2 | abstention < 0,5 |
| H9 petite base | nb de primitives (types) nécessaires pour couvrir toutes les familles représentables du banc (couverture ≥ 0,9) vs nb de familles couvertes | catalogue « une règle par famille » | primitives ≤ 0,5 × familles couvertes | primitives ≥ familles couvertes |

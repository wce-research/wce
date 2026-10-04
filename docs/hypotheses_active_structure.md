# Pré-enregistrement — jalon « Active Structural Discovery » (figé AVANT tout multi-graines / scellé)

Date de gel : 2026-10-03. Critères NON modifiables après observation. Unité : graine (16). IC95 t sur graines.
Récupération fonctionnelle = accord ≥ 0,97 avec la vraie frontière sur la région complète [−3,3]².
Budget de requêtes commun à toutes les méthodes actives : 16. Budget de capacité (H10) : 400 flottants, choisi avant run.
Seuil de fausse réjection visé (H4) : ≤ 0,10 (justification : 5 % est irréaliste avec 8–12 % de bruit et ≤ 16 ancres ;
on accepte 1 bonne règle sur 10 rejetée à tort, contre 0,51 au jalon précédent).

| H | Métrique | Baseline | Effet min. | Succès | Échec |
|---|---|---|---|---|---|
| H1 entrelacement | récup. fonctionnelle (familles held-out + profondeur 3/4) | actif-après-recherche, même budget de requêtes | +0,10 | diff ≥ +0,10 et IC95 > 0 | diff ≤ 0 |
| H2 IG structurel | entropie de l'espace des versions après 8 requêtes | requêtes aléatoires | −20 % | réduction ≥ 20 % de plus que l'aléatoire, ≥ 12/16 graines | ≤ aléatoire |
| H3 clustering fonctionnel | rappel de découverte de familles (held-out) ; fausses promotions | anti-unification AST | +0,2 rappel | rappel ≥ AST + 0,2 ET fausses promotions ≤ AST + 0,5 / graine | rappel ≤ AST |
| H4 ancres sensibles au bruit | fausse réjection (règles correctes) ; détection des fausses règles | ancres binomiales du jalon précédent | — | fausse réjection ≤ 0,10 ET détection ≥ 0,6 (bruit 8 %) | fausse réjection > 0,25 ou détection < 0,4 |
| H5 stabilité sous bruit | taux d'acceptation des programmes fonctionnellement corrects à 8 % et 12 % ; acceptation des non représentables | test de stabilité du jalon précédent | — | acceptation correcte ≥ 0,7 aux deux niveaux ET acceptation non-repr. ≤ 0,35 | acceptation correcte < 0,4 |
| H6 profondeur guidée | évaluations de candidats pour résoudre les familles profondeur 3/4 ; taux résolu | recherche non guidée profondeur 3/4 | −30 % coût | coût ≤ 0,7 × non guidé ET taux résolu ≥ non guidé − 0,05 | coût ≥ non guidé |
| H7 classes canoniques | réutilisation de famille sur instances transformées (rotation, translation, échelle, réflexion) : récupération | familles non canonisées | +0,1 | diff ≥ +0,1 | diff ≤ 0 |
| H8 OOD via expériences structurelles | extrapolation hors région | même système, requêtes aléatoires | +0,03 | diff ≥ +0,03 et IC95 > 0 | diff ≤ 0 |
| H9 résidu à porte | dégradation OOD du résidu vs programme seul ; exactitude en région | résidu global (jalon précédent) | — | dégradation OOD(porte) ≤ 0,5 × dégradation OOD(global) ET région ≥ global − 0,01 | dégradation(porte) ≥ dégradation(global) |
| H10 budget strict | compétences retenues (exactitude ≥ 0,85) sous budget 400 | croissance par experts seuls, même budget | +30 % | retenues ≥ 1,3 × experts-seuls ET 0 dépassement | retenues ≤ experts-seuls |

# Pré-enregistrement des hypothèses — jalon « certificats génératifs » (figé avant toute exécution)

Date de gel : 2026-10-01. Ces critères ne seront PAS modifiés après observation.
Unité statistique : graine (16 graines minimum) ; IC95 par bootstrap/t sur les graines ; comparaisons appariées.

| H | Métrique | Baseline | Succès | Échec |
|---|---|---|---|---|
| H1 certificats génératifs protègent sans stocker les données | rétention finale (flux continu 25 régimes) | raw replay | rétention(gen_cert) ≥ rétention(raw replay) − 0,02 ET 0 octet de données brutes | rétention(gen_cert) < rétention(raw replay) − 0,05 |
| H2 le second ordre attrape des mises à jour dangereuses manquées par le premier ordre | sur des mises à jour projetées premier ordre (J·Δθ≈0) : taux de rejet correct (vraie violation) au second ordre vs premier ordre ; rétention après mise à jour | projection premier ordre | rappel des violations (2nd ordre) − rappel (1er ordre) ≥ 0,2 ET rétention (2nd) > rétention (1er) sur ≥ 12/16 graines | différence de rappel < 0,05 |
| H3 la plasticité compositionnelle reste compétitive en continu bruité | exactitude moyenne finale | raw replay ET croissance structurelle | full ≥ max(raw replay, structural growth) − 0,03 | full < raw replay − 0,05 |
| H4 croissance des paramètres ≪ croissance systématique quand la structure se répète | croissance des paramètres à 100 régimes | croissance structurelle systématique | ratio full / systématique ≤ 0,6 | ratio ≥ 0,9 |
| H5 le coût d'acquisition diminue avec la structure réutilisable | pente (pas d'entraînement par nouvelle compétence) sur l'indice de vie, 100 régimes | pente nulle | pente < 0 avec IC95 entièrement < 0 | IC95 contient 0 ou pente > 0 |
| H6 la pression de capacité se résout par fusion/distillation/élagage sans perte catastrophique | perte d'exactitude moyenne due aux opérations de capacité ; nb d'opérations déclenchées | aucune opération (budget illimité) | ≥ 1 opération de chaque type déclenchée en flux réel ET perte ≤ 0,03 | perte > 0,10 ou aucune opération déclenchée |
| H7 une compétence corrompue se régénère sans instantané | récupération = exactitude régénérée / exactitude avant corruption | régénération du jalon précédent (≈ 0,89 de récupération relative) | récupération médiane ≥ 0,95 | récupération médiane < 0,90 |
| H8 découverte active de lois de transformation | taux d'identification correcte de la loi cachée ; nb d'expériences | expériences aléatoires | identification ≥ 0,8 ET expériences(actives) < expériences(aléatoires) sur ≥ 12/16 graines | identification < 0,5 |

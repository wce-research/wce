# Pré-enregistrement — « Robust Structural Reconstruction & Measurement-Aware Inference » (figé AVANT les runs principaux)

Date de gel : 2026-10-03. AUCUNE modification après le benchmark scellé.
Calcul : AWS SageMaker Processing (eu-north-1), identité WCE-Research-Runner, rôle d'exécution WCE-SageMaker-ExecutionRole.
Graines : **16** (consigné avant run ; 32 seulement si le coût mesuré des 16 reste < 2 h d'instance ml.c5.4xlarge — décision consignée dans le manifeste).
Budget de mesure par tâche : **16 unités de coût** (capteur standard = 1, précis = 3, bon marché = 0,3). Échantillon passif : 120 observations.
Récupération fonctionnelle exacte = accord ≥ 0,97 avec la vraie fonction sur [−3,3]². Récupération de STRUCTURE = squelette canonique identique au vrai.
Budget compact (H9) : ≤ 64 paramètres + ≤ 32 points bruts stockés par compétence.
Référence « jalon précédent » (H7) : système complet ActiveStructuralDiscovery ré-exécuté sur le NOUVEAU banc (même graines).

| H | Métrique | Baseline | Succès | Échec |
|---|---|---|---|---|
| H1 vraisemblance robuste | récupération fonctionnelle, bruit 8–15 % | fit naïf (même candidats, même requêtes = aucune) | +0,10 et IC95 > 0 | diff ≤ 0 |
| H2 séparation structure / bruit | profondeur AST moyenne des structures retenues sur cibles de profondeur 1–2 bruitées ; part « trop profonde » | fit naïf + approfondissement par seuil | profondeur excédentaire réduite ≥ 30 % | non réduite |
| H3 EFA | récupération fonctionnelle (et extrapolation) à budget égal | IG structurel (même inférence) | +0,05 et IC95 > 0 sur au moins une des deux métriques sans perte > 0,02 sur l'autre | ≤ IG structurel sur les deux |
| H4 répétitions adaptatives | récupération à bruit 12–15 % | politique fixe (pas de répétition, même budget) | +0,05 et IC95 > 0 | ≤ 0 |
| H5 calibration P(règle fausse) | ECE (TRAIN+VAL → test) | non calibré | ECE réduit ≥ 50 % | ECE non réduit |
| H6 validation active des familles | précision de promotion (familles promues qui accélèrent / réussissent sur nouvelles instances) | promotion par regroupement (jalon précédent) | précision ≥ baseline + 0,2 | ≤ baseline |
| H7 récupération sous 8–12 % | récupération fonctionnelle | ancien système sur le même banc | +0,10 et IC95 > 0 | ≤ 0 |
| H8 extrapolation conservée | extrapolation hors région | ancien système | ≥ ancien − 0,02 | < ancien − 0,05 |
| H9 budget compact | violations | — | 0 violation | ≥ 1 |
| H10 gain prévu vs réel | corrélation de Spearman entre EFA prévue et gain fonctionnel réalisé | 0 | ρ ≥ 0,3 et IC95 > 0 | ρ ≤ 0 |

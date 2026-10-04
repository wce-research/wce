# WCE Robust Structural Reconstruction & Measurement-Aware Inference — rapport final (jalon gelé)

Campagnes exécutées sur AWS les 2026-10-03 et 2026-10-04, 16 graines, lot scellé évalué une seule fois.
Hypothèses pré-enregistrées avant tout run : `docs/hypotheses_robust_reconstruction.md`.
Synthèse chiffrée : `results/robust_summary.json`. Artefacts bruts : `results/aws/`. Temps : `results/aws_timing.json`.

Bilan : 5 hypothèses soutenues, 3 réfutées, 2 non concluantes.

## 1. Architecture
- Modèle de mesure : chaque observation porte son capteur, son taux de retournement, sa résolution, son poids temporel et sa source.
- Inférence : vraisemblances naïve, à retournement, à contamination et à marge douce ; évidence APPROCHÉE et longueur de description (MDL) ; ensemble de structures compatibles avec réponse « UNKNOWN » possible.
- Mesure active : planificateur séparé de la recherche, six critères (EFA, IG structurel, IG fonctionnel, désaccord, incertitude, aléatoire), choix du capteur, option de répétition.
- Décisions : approfondissement par facteur de Bayes, attribution des écarts en cinq classes, contrôle prédictif a posteriori, calibration de P(règle fausse).
- Monde d'évaluation séparé du code d'inférence, qui ne l'importe jamais.
- Les analogies physiques (Fourier, Boltzmann, Heisenberg, etc.) sont des inspirations de conception, rien de plus.

## 2. Fichiers
- `core/robust_world.py`, `core/robust_reconstruction.py`, `core/robust_experiments.py`, `core/robust_aggregate.py`
- `wce_aws/orchestrate.py`, `wce_aws/sm_entry.py`, `wce_aws/bench.py`
- `tests/test_robust.py`, `docs/hypotheses_robust_reconstruction.md`, cinq scripts de démo dans `scripts/`

## 3. Tests ajoutés
266 tests (objectif : au moins 200) : vraisemblances, postérieure, MDL, gains attendus, planificateur, calibration, fuites, orchestration AWS sans appel réseau, checkpoints et reprise, cache.

## 4. `make check`
Syntaxe OK.

## 5. `make test`
3 828 tests, 7 ignorés, aucune régression.

## 6. Workflow AWS
- Backend EC2 + SSM, toujours explicite (`WCE_BACKEND=ec2`). SageMaker inutilisable (quotas à 0), jamais testé en exécution réelle.
- Pas de SSH ni de clé, instances taguées `Project=WCE`, terminaison automatique, garde-fou d'extinction.
- Unités découpées en sous-unités déterministes ; un checkpoint par sous-unité, publié vers S3 toutes les 45 s.
- Cibles : `aws-login`, `aws-whoami`, `aws-check-guardrails`, `aws-ec2-diagnose`, `aws-progress`, `aws-resume`, `aws-stop`, `aws-inspect`, `aws-timing`, `aws-profile`, `aws-bench`, `aws-fetch-results`, `aws-multiseed`, `aws-run-rest`, `aws-sealed-eval`, `robust-aggregate`.
- Reprise validée de bout en bout (interruption brutale, 55 checkpoints repris, 45 recalculés).
- Écart à la règle Makefile : un seul appel `aws s3 ls` direct, en lecture seule.

## 7. Identité
`WCE-Research-Runner` (profil `wce-runner`, `eu-north-1`). Lectures IAM refusées comme attendu. Aucune permission modifiée par le runner.

## 8. Instances
- Référence scientifique : `c6i.xlarge` (4 vCPU), 2 instances en parallèle (quota de 8 vCPU).
- `t3.large` : test et débogage seulement (1,7 fois plus lent, 1,5 fois plus cher par charge).
- Un processus par vCPU, un thread Torch. Les threads Torch ralentissent (×0,77 à ×0,92) ; les vCPU sont des hyperthreads (59 à 61 % de la vitesse d'un processus seul).

## 9. Temps d'exécution

| Campagne | Sous-unités | Durée par instance |
|---|---|---|
| Banc principal | 1 600 | 1 h 48 et 2 h 12 |
| Bruit, calibration, auxiliaires | 1 676 | 1 h 29 et 1 h 27 |
| Lot scellé | 120 | 7 et 8 min |

CPU moyen 97 à 100 %, mémoire pic 351 Mo par processus. Point chaud : ajustement par gradient (`fit_robust`, 78 % du temps).

## 10. Coût
Environ 1,50 $ pour 8,3 heures-instance sur 17 jobs, essais d'infrastructure compris. Estimation à partir de tarifs à la demande cités de mémoire, à vérifier sur la facture.

## 11. Multi-graines (banc principal, bruit 8 et 12 %, IC 95 %)

| Méthode | Récupération | Structure exacte | Extrapolation | Requêtes |
|---|---|---|---|---|
| Ancien système | 0,117 [0,079 ; 0,155] | 0,141 | 0,778 | 15,6 |
| Passif | 0,047 [0,006 ; 0,088] | 0,055 | 0,619 | 0 |
| Ajustement robuste seul | 0,219 [0,152 ; 0,285] | 0,242 | 0,806 | 0 |
| Postérieure robuste | 0,211 [0,143 ; 0,279] | 0,242 | 0,797 | 0 |
| Actif, IG structurel | 0,328 [0,260 ; 0,396] | 0,250 | 0,847 | 16,0 |
| Actif, EFA | 0,266 [0,207 ; 0,325] | 0,266 | 0,863 | 11,4 |
| Système complet | 0,234 [0,175 ; 0,293] | 0,273 | 0,834 | 10,7 |
| Requêtes aléatoires | 0,266 [0,202 ; 0,329] | 0,242 | 0,838 | 16,0 |
| Requêtes par désaccord | 0,328 [0,260 ; 0,396] | 0,250 | 0,845 | 16,0 |
| Oracle (borne haute) | 0,352 [0,291 ; 0,412] | 0,266 | 0,873 | 16,0 |

Récupération selon la profondeur du monde :

| Méthode | Prof. 1 | Prof. 2 | Prof. 3 | Prof. 4 |
|---|---|---|---|---|
| Ancien système | 0,41 | 0,00 | 0,00 | 0,06 |
| Postérieure robuste | 0,56 | 0,13 | 0,03 | 0,13 |
| Actif, IG structurel | 0,91 | 0,13 | 0,09 | 0,19 |
| Actif, EFA | 0,72 | 0,13 | 0,06 | 0,16 |
| Système complet | 0,66 | 0,13 | 0,06 | 0,09 |
| Oracle | 0,91 | 0,19 | 0,09 | 0,22 |

Reproductibilité : graines 0 et 9 rejouées dans un run séparé, résultats identiques (hors champs de durée).

## 12. Balayage de bruit (récupération)

| Bruit | Système complet | Postérieure robuste | Ancien système | Passif |
|---|---|---|---|---|
| 0 % | 0,479 | 0,333 | 0,292 | 0,188 |
| 2 % | 0,438 | 0,333 | 0,229 | 0,062 |
| 5 % | 0,312 | 0,292 | 0,188 | 0,104 |
| 8 % | 0,250 | 0,208 | 0,208 | 0,104 |
| 12 % | 0,271 | 0,208 | 0,062 | 0,042 |
| 15 % | 0,312 | 0,167 | 0,125 | 0,062 |

Par type de bruit, système complet : 0,385 (frontière), 0,354 (i.i.d.), 0,292 (outliers groupés) ; ancien système : 0,281, 0,135, 0,135.

## 13. Lot scellé (une seule évaluation, 24 tâches dont 20 représentables)

| Méthode | Récupération | Extrapolation |
|---|---|---|
| Système complet | 0,20 [0,01 ; 0,39] | 0,819 |
| Postérieure robuste | 0,15 [−0,02 ; 0,32] | 0,743 |
| Requêtes aléatoires | 0,15 [−0,02 ; 0,32] | 0,792 |
| Ancien système | 0,10 [−0,04 ; 0,24] | 0,687 |
| Passif | 0,00 | 0,713 |

Classement cohérent avec les 16 graines, mais intervalles qui se chevauchent : le lot scellé ne permet pas de conclure seul.

## 14. Calibration de P(règle fausse)
215 cas d'apprentissage (graines 700000+), 348 cas de test ; calibrateurs ajustés sur l'apprentissage seulement.

| Calibrateur | ECE | Brier |
|---|---|---|
| Brut | 0,181 | 0,148 |
| Isotonique | 0,071 | 0,080 |
| Bêta | 0,075 | 0,077 |
| Platt | 0,108 | 0,081 |

## 15. Rappel
- Promotion de familles : rappel 0,688 par validation active, 0,438 par regroupement.
- Récupération sur nouvelle instance : 0,208 avec l'a priori de famille contre 0,313 par recherche complète.

## 16. Extrapolation
Système complet 0,834 contre 0,778 pour l'ancien système (+0,056 [0,013 ; 0,099]). EFA seule : 0,863, meilleure méthode non oracle.

## 17. Fausse promotion et faux rejet
- Familles : 39 promues par validation active, précision 0,846 (6 fausses) ; 23 par regroupement, précision 0,913 (2 fausses).
- Mondes non représentables : le système complet déclare « identifié » dans 28 cas sur 32 ; le test d'identifiabilité classe 13 mondes non représentables sur 16 comme identifiables.
- Réponse « UNKNOWN » : 1,9 % (système complet), 16 % (postérieure robuste), 21 % (oracle).

## 18. Efficacité des requêtes
- EFA : 11,4 requêtes en moyenne ; système complet : 10,7 ; autres politiques actives : 16.
- EFA contre aléatoire : récupération 0,000 [−0,054 ; 0,054], extrapolation +0,025 [0,003 ; 0,047].
- IG structurel et désaccord : meilleures politiques non oracle (0,328).
- Oracle contre système complet : +0,117 [0,072 ; 0,163].

## 19. Complexité
- Taille moyenne des programmes retenus : 1,6 à 2,2 (méthodes robustes), 5,15 (ancien système), 4,82 (passif).
- « Simple + 8 % de bruit » : excès de taille 0,94 (seuil naïf) contre 0,00 (règle bayésienne robuste).
- « Complexe + 0 % de bruit » : excès 0,63 contre 0,00, récupération 0,125 dans les deux cas.

## 20. Capacité
- Aucune violation du budget compact (3,6 paramètres et 17,5 points d'erreur en moyenne).
- Compression du résidu sans apport : 0,911 sans résidu, 0,909 avec résidu brut à porte, 0,831 avec prototypes.

## 21. Hypothèses soutenues
- H2 : excès de profondeur ramené de 0,94 à 0,00.
- H5 : ECE réduit de 0,181 à 0,071.
- H7 : récupération +0,117 [0,055 ; 0,179] par rapport à l'ancien système.
- H8 : extrapolation conservée, +0,056.
- H9 : 0 violation du budget compact.

## 22. Hypothèses réfutées
- H4 : les répétitions adaptatives ne sont jamais choisies (0 sur 64 runs) ; écart nul.
- H6 : validation active moins précise que le regroupement (0,846 contre 0,913).
- H10 : gain prévu par l'EFA sans corrélation avec le gain réalisé (ρ = −0,008 [−0,093 ; 0,078]).

## 23. Hypothèses non concluantes
- H1 : vraisemblance robuste contre naïve, +0,125 [−0,205 ; 0,455].
- H3 : non concluante au sens pré-enregistré, mais défavorable : EFA significativement moins bonne que l'IG structurel en récupération (−0,063 [−0,105 ; −0,020]).

## 24. Bugs corrigés
- Graine du balayage de bruit dépendante de `hash()` d'une chaîne (non reproductible) : corrigée avant tout run.
- Contrôle prédictif des familles : points et étiquettes tirés séparément : corrigé avant tout run.
- Profondeur du monde écrasée par la taille du programme dans les lignes du banc principal : rétablie à l'agrégation, sans recalcul.
- Script EC2 : échec d'installation rapporté « Completed » : corrigé, les échecs remontent.
- Premier pilote sans checkpoint ni progression, arrêté après 52 minutes sans résultat récupérable.
- Non résolu : empreinte de code différente entre la campagne et le run de reproductibilité, cause non identifiée ; résultats identiques.

## 25. Limites
- Récupération faible en absolu : 0,35 pour l'oracle, presque rien au-delà de la profondeur 1.
- Le système ne reconnaît pas qu'un monde est hors de sa classe d'hypothèses.
- Répétitions et choix multi-capteurs sans apport dans ce monde ; le système complet fait moins bien que l'IG structurel seul.
- Évidence approchée, pas évidence bayésienne exacte.
- Monde synthétique en deux dimensions.
- Lot scellé trop petit pour conclure seul.
- SageMaker jamais testé en exécution réelle.

## 26. Prochain goulot
La recherche de structure au-delà de la profondeur 1 : même avec une politique de mesure oracle, la bonne structure n'est presque jamais retrouvée. La politique de mesure n'est pas le facteur limitant ; c'est la génération et l'ajustement des candidats profonds, qui est aussi le point chaud en temps de calcul.

## Gel du jalon
Les résultats de ce jalon ne sont plus modifiés après le commit de gel. Le lot scellé (`SEALED_SHA256` dans `core/robust_world.py`) a été consommé et ne doit plus servir à aucun réglage.

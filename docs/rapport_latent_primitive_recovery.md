# WCE Latent Primitive Recovery — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucune campagne multi-graines, lot scellé jamais
touché. Le pipeline gelé reste la référence officielle ; `flat_scene_k3` est une *confirmed dev baseline candidate*
(trois jeux dev indépendants), pas une baseline confirmatoire ; rien n'est branché par défaut.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

Mise au point faite uniquement sur les clés de composition 1090–1099 (réglage sur 60 tâches, clés 1090–1094).
Évaluation sur **384 tâches jamais utilisées** : clés 1000–1031, profondeurs 1 à 6, régimes standard et rich.

### Expérience principale : trois sources de primitives, un seul aval

| Chaîne | Primitives | Aval |
|---|---|---|
| **A** | atomes du pipeline gelé | aval fort |
| **B** | `flat_scene_k3` | aval fort |
| **C** | `LatentPrimitiveSetRecovery` | aval fort |

Aval fort, identique pour les trois : hypothèses de primitives → ensembles compatibles (utilité conditionnelle,
sauvetage par paires, trois meilleurs ensembles) → composition sur classes d'équivalence par ensemble → finalisation
validée. Aucune chaîne ne reçoit d'oracle ; six atomes au plus par ensemble.

C ajoute, en amont et autour de cet aval : un vivier « rappel d'abord » (cinq paramétrisations par type, plus les
hypothèses issues des fragments et des composantes de consensus, sans élimination), la réinjection des primitives
ajustées, et une boucle bornée (deux reprises au plus) qui propose des primitives sur le RÉSIDU de la composition ;
une reprise n'est gardée que si les erreurs sur les observations baissent.

Configuration figée avant le lancement (`results/deep_configs/lpr.json`) : gain minimal 1 ; la boucle est toujours
tentée. En réglage, les six configurations essayées donnent un accord entre 0,899 et 0,902.

Bras secondaire pré-enregistré : **B5** (`flat_scene_k5`, même aval). Variantes de C, secondaires : sans boucle, sans
paires, sans réinjection, `k = 3`, vivier `flat` seul. Chaînes historiques rapportées pour la continuité : `A_full`
(pipeline gelé complet) et `B_full` (`flat_scene_k3` + aval actuel). Aucun autre bras ne sera ajouté.

### Métriques primaires

Toutes les vraies primitives présentes (dans le vivier d'hypothèses, et dans l'ensemble retenu) ; erreur de paramètre
conditionnelle ; structure candidate ≥ 0,97 ; final ≥ 0,99 ; accord final ; calcul.

Une vraie primitive est « présente » si une primitive candidate la reproduit à 0,95 d'accord fonctionnel (« strict » :
0,99 ; « exact » : même type et paramètres dans la tolérance). Les trois niveaux sont rapportés séparément.

### Hypothèses

| # | Hypothèse | Critère |
|---|---|---|
| LP1 | C améliore le rappel d'ENSEMBLE | part des tâches où toutes les vraies primitives sont présentes dans le vivier : C − B et C − A, IC 95 % > 0 |
| LP2 | C améliore l'aval | C − B et C − A sur la structure candidate ET sur l'accord final, IC > 0 |
| LP3 | le régime « rappel d'abord » augmente le rappel individuel | rappel moyen dans le vivier, C − B, IC > 0 |
| LP4 | les faux positifs restent tolérables | vrais + 10 faux : récupération à moins de 0,05 des vrais seuls |
| LP5 | le défaut de réduction est corrigé | vrais atomes seuls : tous gardés, sélection d'ensembles − ancienne réduction, IC > 0 |
| LP6 | plusieurs paramétrisations par type aident | B5 − B sur le rappel d'ensemble et sur l'accord, IC > 0 |
| LP7 | le résidu de composition retrouve une primitive manquante | contrôlé, un vrai atome retiré : retrouvé dans ≥ 50 % des tâches, et récupération après > avant |
| LP8 | la boucle aide en conditions réelles | C − `C_noloop` sur l'accord, IC > 0 |
| LP9 | les manques ne sont pas indépendants | corrélation positive entre manques d'une même tâche ; écart au modèle `rappel^n` |
| LP10 | le calcul reste compact | C sous 5 s par tâche, et sous `A_full` et `B_full` |
| LP11 | condition de campagne | C bat A, B, `A_full` et `B_full` sur l'accord final, IC > 0 |

La courbe de sensibilité (erreur de paramètre → récupération, par type et par famille de paramètres) et la tolérance
aux faux atomes (+1, +3, +5, +10) sont descriptives. L'exploration « ensemble incomplet ou monde hors classe » est
exploratoire et ne sera pas présentée comme un résultat confirmatoire.

Attente déclarée d'après la mise au point : accord de C de 0,902 sur 60 tâches, du même ordre que la chaîne du
sous-jalon précédent ; LP2 et LP11 peu probables ; LP5 et LP7 probables (vérifiées sur des cas construits) ; LP8
probablement nulle. Le rappel d'ensemble n'a pas pu être lu en mise au point (défaut de mesure corrigé avant le run).

### Règle de verdict et décision de campagne, fixées d'avance

- `READY FOR 16-SEED CONFIRMATORY CAMPAIGN` : LP1, LP2, LP10 et LP11 toutes soutenues. Alors seulement, proposer la
  campagne — sans la lancer.
- `LATENT PRIMITIVE RECOVERY IMPROVED BUT INSUFFICIENT` : LP1 soutenue (le rappel d'ensemble progresse), mais LP2 ou
  LP11 non soutenue. Améliorer le seul rappel individuel (LP3) ne suffit pas à ce verdict.
- `LATENT PRIMITIVE RECOVERY STILL BLOCKING` : LP1 non soutenue.

Dans les deux derniers cas : identifier le nouveau verrou et s'arrêter. Le lot scellé n'est lancé dans aucun cas.

## 1. Protocole exécuté

Run AWS `wce-deeplp-20261010-040904` (EC2 + SSM, `eu-north-1`, 2 × `c6i.xlarge`, 432 sous-unités par instance, 0 échec,
instances terminées). 18 760 lignes agrégées, 0 unité en erreur (`results/deep_latent_primitive_summary.json`).
384 tâches dev jamais vues (clés 1000–1031, profondeurs 1–6, 64 tâches par profondeur). Aucun bras ajouté après le
pré-enregistrement. Différences appariées par tâche, IC 95 % bootstrap. Une seule graine de tâches par clé : rien
ici n'est confirmatoire.

## 2. Comparaison principale (même aval fort)

| Chaîne | Accord final | Accord d4–6 | Final ≥ 0,99 | Candidate ≥ 0,97 | Rappel vivier | Toutes présentes (vivier) | Au plus une absente | Toutes gardées | Erreur param. | Hypothèses | Temps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (gelé) | 0,895 | 0,840 | 0,075 | 0,297 | 0,413 | 0,242 | 0,354 | 0,198 | 0,134 | 36,5 | 28,3 s |
| B (`flat_scene_k3`) | 0,896 | 0,844 | 0,094 | 0,242 | 0,427 | 0,224 | 0,341 | 0,164 | 0,065 | 14,2 | 2,8 s |
| B5 (`flat_scene_k5`) | 0,898 | — | 0,091 | 0,268 | 0,477 | 0,245 | — | — | — | 23,3 | 3,3 s |
| **C (LPSR)** | 0,902 | 0,858 | 0,088 | 0,315 | 0,546 | 0,271 | 0,451 | 0,195 | 0,067 | 45,2 | 9,0 s |
| `A_full` | 0,883 | — | 0,115 | 0,320 | — | — | — | — | — | — | 26,1 s |
| `B_full` | 0,902 | — | 0,151 | 0,378 | — | — | — | — | — | — | 20,6 s |

Différences appariées (toutes tâches) :

| Différence | Accord final | Candidate ≥ 0,97 | Toutes présentes (vivier) | Rappel vivier |
|---|---|---|---|---|
| C − A | +0,008 [+0,002 ; +0,014] | +0,018 [−0,010 ; +0,050] | +0,029 [−0,003 ; +0,060] | +0,133 [+0,102 ; +0,162] |
| C − B | +0,007 [+0,001 ; +0,013] | +0,073 [+0,044 ; +0,104] | +0,047 [+0,026 ; +0,068] | +0,119 [+0,100 ; +0,139] |
| C − B5 | +0,005 [−0,001 ; +0,010] | — | +0,026 [+0,010 ; +0,042] | — |
| C − `A_full` | +0,020 [+0,013 ; +0,026] | — | — | — |
| C − `B_full` | +0,001 [−0,005 ; +0,006] | −0,063 [−0,094 ; −0,034] | — | — |

Sur « final ≥ 0,99 », C ne bat personne : C − B −0,005 ; C − `A_full` −0,026 ; C − `B_full` −0,063 [−0,091 ; −0,034].
Aux profondeurs 4–6, l'accord de C dépasse A (+0,018 [+0,009 ; +0,027]), B (+0,014 [+0,005 ; +0,023]) et `B_full`
(+0,010 [+0,001 ; +0,018]). B − A sur l'accord : +0,001, non significatif.

Variantes de C (toutes tâches, accord) : sans boucle 0,900 ; sans paires 0,904 ; sans réinjection 0,900 ; `k = 3`
0,905 ; vivier `flat` seul 0,903. Aucune ne diffère de C sur l'accord (IC incluant 0). Les sources fragments et
composantes ajoutent du rappel individuel (+0,024 [+0,009 ; +0,039]) sans effet sur l'ensemble ni sur l'aval.

## 3. Loi d'échelle du rappel d'ensemble

| Atomes | Rappel d'atome B | Toutes présentes B | `rappel^n` B | Rappel d'atome C | Toutes présentes C | `rappel^n` C | Au plus une absente C | Toutes gardées C | Visibilité moy. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,984 | 0,984 | 0,984 | 0,984 | 0,984 | 0,984 | 1,000 | 0,828 | 1,00 |
| 2 | 0,531 | 0,344 | 0,282 | 0,680 | 0,484 | 0,462 | 0,875 | 0,313 | 0,85 |
| 3 | 0,365 | 0,016 | 0,049 | 0,495 | 0,125 | 0,121 | 0,484 | 0,031 | 0,74 |
| 4 | 0,281 | 0,000 | 0,006 | 0,453 | 0,031 | 0,042 | 0,297 | 0,000 | 0,73 |
| 5 | 0,209 | 0,000 | 0,000 | 0,344 | 0,000 | 0,005 | 0,047 | 0,000 | 0,67 |
| 6 | 0,190 | 0,000 | 0,000 | 0,320 | 0,000 | 0,001 | 0,000 | 0,000 | 0,68 |

Chaîne A : rappel d'atome 1,000 / 0,617 / 0,349 / 0,203 / 0,184 / 0,125 ; toutes présentes 1,000 / 0,438 / 0,016 / 0 /
0 / 0. Le rappel d'ensemble suit de près le modèle indépendant `rappel^n` pour toutes les chaînes. Deux effets se
multiplient : le rappel par atome chute lui-même avec la profondeur (C : 0,98 → 0,32), puis il est élevé à la
puissance n.

Corrélation des manques entre atomes d'une même tâche (2 240 paires) : phi = +0,064 (A), −0,016 (B), +0,008 (B5),
+0,050 (C). P(manque | autre manqué) est légèrement INFÉRIEURE à P(manque) : 0,611 contre 0,625 pour C. Paires de même
type (370) : phi +0,146 (A), −0,007 (B), +0,176 (C) ; types différents : +0,048, −0,019, +0,026. Pas d'IC calculé sur
phi.

Rappel par type (vivier) :

| Type | A | B | B5 | C |
|---|---|---|---|---|
| demi-plan | 0,522 | 0,430 | 0,530 | 0,661 |
| bande | 0,315 | 0,402 | 0,442 | 0,504 |
| disque | 0,188 | 0,337 | 0,396 | 0,525 |
| anti-disque | 0,151 | 0,247 | 0,307 | 0,360 |
| anneau | 0,158 | 0,213 | 0,225 | 0,263 |
| secteur | 0,276 | 0,181 | 0,238 | 0,319 |

Rappel selon la visibilité de l'atome (part de sa frontière visible dans la scène) :

| Visibilité | n | B | C |
|---|---|---|---|
| 0,0–0,2 | 68 | 0,029 | 0,059 |
| 0,2–0,4 | 186 | 0,043 | 0,134 |
| 0,4–0,6 | 214 | 0,084 | 0,234 |
| 0,6–0,8 | 212 | 0,269 | 0,420 |
| 0,8–1,0 | 664 | 0,494 | 0,642 |

Même un atome presque entièrement visible n'est retrouvé que 64 % du temps par C. Erreur de paramètre conditionnelle
de C, par type : secteur — sommet 0,462, direction 0,181, demi-ouverture 0,136 ; disque — centre 0,098, rayon 0,049 ;
anti-disque — centre 0,105, rayon 0,053 ; demi-plan — orientation 0,049, décalage 0,073 ; bande — 0,015 / 0,040 /
0,057 ; anneau — centre 0,039, rayons 0,046 / 0,023.

## 4. Sensibilité : erreur de paramètre → récupération

Vrais atomes perturbés, tâches où les vrais atomes non perturbés donnent une structure fonctionnelle (n = 372).

| Perturbation σ | Final ≥ 0,99 | Final ≥ 0,97 | Accord |
|---|---|---|---|
| 0,02 | 0,307 | 0,850 | 0,982 |
| 0,05 | 0,065 | 0,347 | 0,954 |
| 0,10 | 0,016 | 0,118 | 0,917 |
| 0,20 | 0,005 | 0,030 | 0,852 |

Par famille (final ≥ 0,99 à σ = 0,02 / 0,05 / 0,10) : orientation 0,61 / 0,25 / 0,15 ; centre 0,76 / 0,36 / 0,15 ;
décalage 0,83 / 0,44 / 0,21 ; rayon 0,83 / 0,39 / 0,22 ; largeur 0,85 / 0,61 / 0,36. Par type, à σ = 0,05 : bande 0,17 ;
anneau 0,27 ; secteur 0,35 ; demi-plan 0,48 ; disque 0,53 ; anti-disque 0,62. L'orientation est la famille la plus
fragile, la bande le type le plus fragile.

L'erreur de paramètre moyenne des primitives retrouvées par C (0,067) se situe au-delà de σ = 0,05, là où la
récupération à 0,99 tombe à 6 %.

## 5. Faux positifs et correction de la réduction

Vrais atomes plus k faux, sélection d'ensembles :

| Faux ajoutés | Accord | Final ≥ 0,99 | Tous gardés | Faux gardés | Final ≥ 0,99 (d4–6) |
|---|---|---|---|---|---|
| 0 | 0,995 | 0,945 | 0,943 | 0,00 | 0,891 |
| 1 | 0,994 | 0,932 | 0,940 | 0,01 | 0,865 |
| 3 | 0,997 | 0,951 | 0,938 | 0,06 | 0,906 |
| 5 | 0,997 | 0,922 | 0,919 | 0,19 | 0,859 |
| 10 | 0,990 | 0,854 | 0,857 | 0,52 | 0,750 |

Appariées contre +0 : +1 final −0,013 [−0,029 ; +0,003] ; +3 +0,005 [−0,010 ; +0,021] ; +5 −0,023 [−0,047 ; −0,003] ;
+10 −0,091 [−0,125 ; −0,060], accord −0,005 [−0,011 ; +0,001].

Correction de la réduction (vrais atomes seuls, sélection d'ensembles − ancienne réduction gloutonne) : tous gardés
+0,141 [+0,104 ; +0,177] (0,802 → 0,943) ; rappel gardé +0,117 [+0,089 ; +0,145] ; final ≥ 0,99 +0,156
[+0,117 ; +0,195] (0,789 → 0,945) ; accord +0,052 [+0,034 ; +0,073]. En moyenne 1,44 vrai atome par tâche (2,48 à
d4–6) n'a d'utilité que conditionnelle : c'est ce que l'ancienne réduction supprimait.

Avec des primitives DÉCOUVERTES, en revanche, la sélection ne garde pas ce qui est dans le vivier : pour C, toutes
présentes dans le vivier 0,271 mais toutes gardées 0,195, et 4,4 des 5,2 atomes gardés sont faux.

## 6. Complétude, primitive manquante, boucle

Contrôle (vrais atomes, puis un vrai atome retiré) :

| Condition | n | Signalé incomplet | Accord | Final ≥ 0,99 |
|---|---|---|---|---|
| ensemble complet | 384 | 0,096 | 0,995 | 0,948 |
| un atome retiré, avant boucle | 320 | 0,944 | 0,826 | 0,022 |
| un atome retiré, après boucle | 320 | — | 0,915 | 0,169 |

L'atome retiré est retrouvé par le résidu dans 45,3 % des tâches et gardé dans 39,1 % (2,7 itérations). Par
profondeur : 0,70 (d2), 0,56 (d3), 0,47 (d4), 0,31 (d5), 0,22 (d6). Par type : disque 0,60 ; bande 0,58 ; anneau 0,52 ;
demi-plan 0,50 ; secteur 0,40 ; **anti-disque 0,00** (42 tâches). Dans ce contrôle, le signal de complétude sépare bien
complet (9,6 % signalés) et incomplet (94,4 %).

En conditions réelles (chaîne C) : la boucle tourne dans 90,6 % des tâches (2,4 itérations) ; C − `C_noloop` : accord
+0,002 [−0,001 ; +0,006], candidate ≥ 0,97 +0,039 [+0,016 ; +0,065], toutes présentes +0,013 [+0,003 ; +0,026], rappel
+0,031 [+0,022 ; +0,041], temps +4,6 s. L'état `COMPLETE` est déclaré pour 346 tâches sur 384, alors que toutes les
primitives n'y sont présentes que dans 29,8 % des cas : l'état n'est pas un indicateur fiable de complétude avec des
primitives découvertes.

Le rappel d'ensemble paie-t-il en aval ? Oui, fortement, mais il est rare. Final ≥ 0,99 quand toutes les vraies
primitives sont gardées (niveau strict) contre sinon : A 0,528 (n = 36) contre 0,029 ; B 0,778 (n = 18) contre 0,060 ;
C 0,560 (n = 25) contre 0,056. Accord : 0,985–0,989 contre 0,885–0,897.

Hors classe (exploratoire, 32 tâches par cas) : primitive absente → 32 `COMPLETE` sur 32 ; relation absente → 29
`COMPLETE`, 3 `INCOMPLETE` ; composition absente → 27 / 5. `POSSIBLY_OUT_OF_CLASS` n'est jamais déclaré. La détection
hors classe ne fonctionne pas.

## 7. Hypothèses pré-enregistrées

| # | Résultat | Statut |
|---|---|---|
| LP1 | C − B +0,047 [+0,026 ; +0,068] ; C − A +0,029 [−0,003 ; +0,060] | **non soutenue** (les deux IC étaient exigés) |
| LP2 | candidate : C − B +0,073, IC > 0 ; C − A +0,018 [−0,010 ; +0,050] ; accord : les deux IC > 0 | **non soutenue** |
| LP3 | rappel individuel C − B +0,119 [+0,100 ; +0,139] | soutenue |
| LP4 | +10 faux : final ≥ 0,99 −0,091 [−0,125 ; −0,060] ; accord −0,005 | **non soutenue** à +10 ; tenue jusqu'à +5 (−0,023) |
| LP5 | tous gardés +0,141 [+0,104 ; +0,177] | soutenue |
| LP6 | B5 − B : ensemble +0,021 [+0,008 ; +0,037] ; accord +0,002, IC incluant 0 | **non soutenue** (partielle) |
| LP7 | retrouvé 45,3 % (< 50 %) ; après > avant (0,169 contre 0,022) | **non soutenue** (partielle) |
| LP8 | C − `C_noloop` accord +0,002 [−0,001 ; +0,006] | **non soutenue** |
| LP9 | phi entre −0,02 et +0,06 ; rappel d'ensemble ≈ `rappel^n` | **non soutenue** |
| LP10 | 9,0 s (> 5 s) ; inférieur à `A_full` 26,1 s et `B_full` 20,6 s | **non soutenue** |
| LP11 | bat A, B, `A_full` ; C − `B_full` +0,001 [−0,005 ; +0,006] | **non soutenue** |

Attentes déclarées : LP2 et LP11 peu probables — confirmé ; LP8 nulle — confirmé ; LP5 probable — confirmé ; LP7
probable — démentie sur le seuil de 50 %.

## 8. Réponses aux dix questions

**1. Pourquoi le rappel d'ensemble s'effondre-t-il avec la profondeur ?** Par deux effets multiplicatifs. Le rappel
par atome baisse avec la profondeur (C : 0,98 à un atome, 0,45 à quatre, 0,32 à six), et l'ensemble exige tous les
atomes à la fois, ce qui élève ce rappel à la puissance n. Le modèle indépendant reproduit les valeurs observées
(C, trois atomes : 0,125 observé, 0,121 prédit). La baisse par atome est liée à la visibilité (0,06 sous 20 % de
frontière visible, 0,64 au-dessus de 80 %), mais la visibilité n'explique pas tout : un atome presque entièrement
visible est encore manqué une fois sur trois.

**2. Les erreurs de primitives sont-elles indépendantes ?** À peu près. Phi entre −0,02 et +0,06 selon la chaîne ;
manquer un atome ne rend pas l'autre plus susceptible d'être manqué. Seule dépendance visible : les paires de même
type (phi +0,15 à +0,18 pour A et C, nul pour B), sans IC. Il n'y a donc pas de cause commune simple à corriger : il
faut relever le rappel par atome lui-même.

**3. Une stratégie recall-first aide-t-elle ?** Sur le rappel individuel, oui (+0,119 contre B, +0,133 contre A).
Sur le rappel d'ensemble, oui contre B (+0,047), pas démontré contre A. Au-delà de trois atomes, non : toutes
présentes 0,031 à quatre atomes, 0 à cinq et six. Et le gain du vivier se perd à la sélection (0,271 dans le vivier,
0,195 gardées).

**4. Les faux positifs restent-ils tolérables ?** Avec les vrais atomes présents, oui jusqu'à +5 (final ≥ 0,99
−0,023), non à +10 (−0,091 ; −0,141 à d4–6), l'accord restant à 0,990. Avec des primitives découvertes, le problème
est d'un autre ordre : 4,4 des 5,2 atomes gardés sont faux, parce que les vrais manquent ou sont imprécis.

**5. Top-k paramètres aide-t-il ?** Peu. `k = 5` contre `k = 3` : rappel d'ensemble +0,021 [+0,008 ; +0,037], accord
+0,002 non significatif. Dans C, `k = 3` fait aussi bien que `k = 5`.

**6. La composition peut-elle guider la recherche de primitives manquantes ?** En contrôle, partiellement : quand un
seul vrai atome manque et que les autres sont exacts, le résidu le retrouve dans 45 % des tâches (70 % à deux
atomes, 22 % à six) et le final ≥ 0,99 passe de 0,022 à 0,169. L'anti-disque n'est jamais retrouvé (0 sur 42) — cause
non établie.

**7. La boucle itérative récupère-t-elle des atomes manquants ?** En conditions réelles, marginalement : rappel
+0,031, toutes présentes +0,013, structure candidate +0,039, mais accord final +0,002 non significatif, pour un
temps doublé (4,5 s → 9,0 s). Le résidu est construit sur une composition déjà fausse, à plusieurs atomes manquants.

**8. Le set recall améliore-t-il réellement la structure finale ?** Quand il est atteint, oui : final ≥ 0,99 de
0,53–0,78 contre 0,03–0,06 selon que l'ensemble est complet au niveau strict ou non. Mais le gain de rappel d'ensemble
obtenu ici (+0,03 à +0,05, concentré sur un à trois atomes) ne se traduit qu'en +0,007 d'accord et en aucun gain de
final ≥ 0,99.

**9. Le coût reste-t-il compact ?** 9,0 s par tâche et 45 hypothèses : au-dessus du seuil pré-enregistré de 5 s,
mais sous `A_full` (26,1 s) et `B_full` (20,6 s). La moitié du coût est la boucle, qui n'apporte rien de mesurable
sur l'accord.

**10. Le système est-il enfin prêt pour campagne confirmatoire ?** Non. LP1, LP2, LP10 et LP11 sont toutes non
soutenues. La campagne 16 graines n'est pas proposée ; le lot scellé n'a pas été touché.

## 9. Limites

- Partition dev, une seule graine de tâches par clé, 64 tâches par profondeur ; les IC par profondeur sont larges.
- LP1 échoue sur un IC qui frôle zéro (C − A +0,029 [−0,003 ; +0,060]) ; la règle écrite d'avance est appliquée
  telle quelle, sans relecture.
- « Présente » repose sur un accord fonctionnel à 0,95 mesuré sur le domaine ; au niveau strict (0,99) les parts
  sont deux à trois fois plus basses (C 0,091, B 0,057, A 0,120 — A devant C).
- Phi sans intervalle de confiance ; corrélation mesurée par paires, pas d'analyse d'ordre supérieur.
- Sensibilité et faux positifs partent des VRAIS atomes : ils bornent l'aval, ils ne décrivent pas la chaîne réelle.
- Hors classe : 32 tâches par cas, exploratoire, résultat négatif.
- `B_full` reste égal à C sur l'accord et meilleur sur final ≥ 0,99 et structure candidate : l'aval fort commun à
  A, B, C n'est pas le meilleur aval connu pour `flat_scene_k3`.
- Le zéro de l'anti-disque dans la recherche par résidu n'est pas expliqué ; il peut venir de la mesure
  (complément retrouvé sous un autre type) autant que de la méthode. `POST-HOC / EXPLORATORY`, non vérifié.

## 10. Nouveau verrou

Le verrou n'est pas la corrélation des erreurs, ni les faux positifs, ni la réduction (corrigée), ni la composition
(0,95 de final ≥ 0,99 à atomes exacts). C'est le **rappel et la précision par atome en scène composée** : 0,32 à 0,45
aux profondeurs 4–6, avec une erreur de paramètre (0,067) située dans la zone où la sensibilité fait tomber la
récupération à 0,99 sous 10 %. Élargir le vivier ne suffit pas : il faudrait un rappel par atome d'environ 0,9 pour
obtenir 50 % d'ensembles complets à six atomes. Second point : la sélection d'ensembles perd des vrais atomes
présents dans le vivier quand ils sont imprécis.

## 11. Verdict

`LATENT PRIMITIVE RECOVERY STILL BLOCKING`

Règle pré-enregistrée : LP1 non soutenue. Le rappel individuel progresse nettement (LP3) et le rappel d'ensemble
progresse contre `flat_scene_k3`, mais pas de façon démontrée contre le pipeline gelé, et il reste nul au-delà de
quatre atomes. Aucune campagne 16 graines proposée, lot scellé intact, rien branché par défaut ; `flat_scene_k3`
reste *confirmed dev baseline candidate*.

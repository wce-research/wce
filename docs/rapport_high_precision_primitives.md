# WCE High-Precision Primitive Estimation — rapport (dev, exploratoire)

Statut : sous-jalon exploratoire, partition **dev** uniquement. Aucune campagne multi-graines, lot scellé jamais
touché. Le pipeline gelé reste la référence officielle ; `flat_scene_k3` reste *confirmed dev baseline candidate* ;
rien n'est branché par défaut.

## 0. Pré-enregistrement (écrit avant le lancement des runs AWS)

### 0.1 Ce que la mise au point a montré (clés 1090–1095 uniquement, petits effectifs, `POST-HOC / EXPLORATORY`)

Ces constats ont décidé de la conception. Ils portent sur 24 à 36 tâches de mise au point et ne sont pas des
résultats ; ils sont repris, à l'échelle, par les expériences pré-enregistrées ci-dessous.

1. **Les objectifs locaux à une seule primitive ne sont pas alignés avec la fonction.** En partant de la VRAIE
   primitive et en montant sur le contraste de paires voisines, on s'en éloigne (erreur relative médiane 0 → 0,10 à
   forte visibilité, 0 → 0,37 à faible visibilité) ; l'objectif trouvé dépasse celui de la vérité dans plus de 90 %
   des cas. Même constat pour un plancher d'erreurs sur une grille spatiale.
2. **L'objectif conjoint « plancher par cellules » est aligné.** Les vrais atomes perturbés de 0,10 sont ramenés à
   une erreur relative médiane d'environ 0,05 jusqu'à quatre atomes. À six atomes et 300 observations, il
   sur-ajuste (64 cellules) : une validation hors-échantillon est nécessaire.
3. **Une recherche gloutonne directe sur le plancher, sans vivier, échoue** dès deux atomes (le premier atome choisi
   approxime la composition entière). Abandonnée.
4. **L'extraction des courbes de frontière est le maillon faible, et il est algorithmique.** Avec 3 000 observations
   sans bruit, les droites et cercles existants ne retrouvent que 35 à 50 % des courbes vraies à six atomes. Un
   extracteur à votes avec normales et consensus séquentiel monte à 0,91–1,00 (droites) et 0,73–0,80 (cercles) à
   deux et quatre atomes dans cette condition. Aux densités des tâches (300 et 600 observations bruitées), les deux
   extracteurs sont au même niveau, bas (0,1 à 0,5).
5. **Sur 24 tâches standard de mise au point, la nouvelle chaîne ne fait pas mieux que le même aval sans elle**
   (accord 0,87 contre 0,88). Sur 9 tâches denses, la précision atomique monte nettement (atomes à moins de 0,05
   d'erreur relative : 0,33 → 0,70) sans gain visible sur l'accord final.

### 0.2 Méthode primaire : chaîne C

`observations → vivier → ensembles compatibles → amélioration validée → même aval fort`

- **Vivier, rappel d'abord** : vivier existant (cinq paramétrisations par type, composantes de consensus) ET
  estimateurs spécialisés par famille — droites et cercles par votes avec normales locales, bande adaptée à
  l'espacement des observations, consensus séquentiel ; puis assemblage : demi-plan = une droite ; disque et
  anti-disque = un cercle ; bande = deux droites parallèles à orientation partagée ; anneau = deux cercles à centre
  partagé ; secteur = deux demi-droites de même sommet, chaque droite devant porter des éléments du côté de sa
  demi-droite. Budget : 12 droites, 10 cercles, cinq hypothèses au plus par type composé.
- **Amélioration validée des trois meilleurs ensembles** : raffinement conjoint local sur le plancher par cellules
  (point initial conservé), puis au plus deux naissances de primitive par gain de plancher conditionnel, orientées par
  la morphologie du résidu (linéaire, arc, paire parallèle, paire concentrique, radiale), puis élagage conditionnel.
  Validated Transformation Principle : la procédure entière est rejouée sur trois sous-échantillons et jugée sur les
  observations qu'elle n'a pas vues ; elle n'est retenue que si elle y corrige significativement plus d'erreurs
  qu'elle n'en crée (test apparié, z ≥ 1,64) ; sinon l'ensemble initial est rendu tel quel. Ce seuil a été ajouté
  avant le lancement : sans lui, la procédure était acceptée sur des étiquettes aléatoires (test unitaire).
- Les primitives améliorées **rejoignent** le vivier ; l'aval fort (ensembles → composition sur classes
  d'équivalence → finalisation validée), identique pour A, B et C, choisit.

Configuration figée : `results/deep_configs/hp.json`. Aucun composant appris n'est introduit : aucun problème précis
n'a été identifié que les estimateurs explicites ne puissent attaquer, et le précédent proposeur neuronal n'apportait
rien hors distribution. Le secteur de ce monde est un coin non borné (sommet, direction, demi-ouverture) : il n'a ni
rayon ni arc.

### 0.3 Données

Clés **1100–1131**, jamais utilisées. Profondeurs 1 à 6. Régimes `standard` (300 observations, 5 %) et `rich`
(600, 2 %) : régimes **primaires**. Régime `dense` (2 400 observations, 2 %), mêmes mondes : régime **secondaire**,
pour séparer manque d'information et défaut d'algorithme. Régime `hard` (160, 10 %), jamais utilisé en mise au
point : condition **hors distribution**. Banc par primitive : scènes masquées nouvelles (indices 21100–21131).

### 0.4 Expériences et bras

| Expérience | Contenu |
|---|---|
| `chain` | A atomes du pipeline gelé ; B `flat_scene_k3` ; C — même aval fort. Secondaire pré-enregistré : B5. Variantes de C : vivier seul (`C_pool`), sans estimateurs spécialisés, sans raffinement, sans naissance, validation optimiste (`C_cv`), `k = 1`, `k = 3`, courbes indépendantes (`C_nojoint`). Historiques (standard, rich) : `A_full`, `B_full`, `C_prev` (chaîne du sous-jalon précédent). |
| `single` | Une primitive, type connu : 6 types × densité (150, 300, 600, 2 400) × bruit (0 ; 5 %) × masquage (0 ; 0,3 ; 0,6). Méthodes : gradient générique sur la cible, gradient générique sur le contraste, inférence existante, courbes spécialisées, courbes indépendantes (bande, anneau), recherche directe de la primitive entière, raffinement, consensus (médiane de rééchantillonnages), oracles partiels (centre, orientation, taille), raffinement depuis la vérité (plancher d'information), identifiabilité de la vraie primitive. |
| `sens` | Un paramètre d'un vrai atome perturbé de 0,02 / 0,05 / 0,10 / 0,20, les autres atomes exacts, sans finalisation : dégradation fonctionnelle de l'atome, puis récupération finale. |
| `birth` | Un vrai atome retiré (contexte exact, et contexte perturbé de 0,05) : naissance nouvelle, sans morphologie, non validée, contre l'ancienne recherche sur le résidu ; fausses naissances sur ensembles complets ; trois mesures de « retrouvé » dont une indépendante de la polarité. |
| `ood` | B, C, `C_pool` au régime `hard`. |

Aucun autre bras ne sera ajouté. Aucun `k` autre que 1, 3, 5.

### 0.5 Métriques

- **Rappel individuel** (continuité avec le sous-jalon précédent) : vraie primitive reproduite à 0,95 d'accord
  fonctionnel par une hypothèse du vivier.
- **Précision** : erreur fonctionnelle RELATIVE (désaccord rapporté à l'aire de la primitive, polarité libre) ≤ 0,10
  (« précise ») et ≤ 0,05 (« fine ») ; erreurs par famille de paramètres ; part des atomes dans la tolérance de leur
  famille.
- **Rappel d'ensemble** : toutes les vraies primitives présentes (0,95), par nombre d'atomes ; version précise
  (toutes à ≤ 0,10).
- Structure candidate ≥ 0,97 ; accord final ; final ≥ 0,99 ; accord en extrapolation.
- Coût : temps par tâche, évaluations de plancher, hypothèses, mémoire résidente maximale, coût AWS.

**Tolérance par famille** (règle mécanique, fixée ici) : pour chaque type et famille de paramètres, la tolérance est
la plus grande perturbation testée telle que, pour elle et pour toutes les plus petites, la récupération finale
≥ 0,99 reste au moins égale à 80 % de celle sans perturbation (tâches dont les vrais atomes non perturbés donnent une
structure fonctionnelle) ; « < 0,02 » si aucune. Calculée sur l'expérience `sens` seule ; aucun choix de méthode n'en
dépend.

Différences appariées par tâche, IC 95 % bootstrap. Régimes primaires réunis sauf mention.

### 0.6 Hypothèses

| # | Hypothèse | Soutenue si | Réfutée si |
|---|---|---|---|
| HP1 | C augmente le rappel individuel | C − B et C − A, IC > 0, ET rappel de C aux profondeurs 4–6 ≥ 0,70 | un IC inclut 0, ou rappel d4–6 < 0,60 |
| HP2 | C augmente la précision | part d'atomes précis (≤ 0,10) : C − B et C − A, IC > 0 | un IC inclut 0 |
| HP3 | C augmente le rappel d'ensemble | toutes présentes : C − B et C − A, IC > 0, ET ≥ 0,25 aux profondeurs 4–6 | un IC inclut 0, ou < 0,10 à d4–6 |
| HP4 | C améliore la structure candidate | candidate ≥ 0,97 : C − B et C − A, IC > 0 | un IC inclut 0 |
| HP5 | le gain traverse jusqu'au final | accord final ET final ≥ 0,99 : C − B et C − A, IC > 0 | un IC inclut 0 sur l'une des deux mesures |
| HP6 | les estimateurs spécialisés battent le gradient générique | banc `single`, part précise au premier rang : courbes − gradient, IC > 0 ; rapporté par type | IC inclut 0 ou < 0 |
| HP7 | plusieurs paramétrisations aident | banc : vérité dans les 5 premiers − au premier rang > 0,05 ; chaîne : C − `C_k1` sur l'accord, IC > 0 | l'un des deux manque |
| HP8 | le consensus réduit les erreurs catastrophiques | banc : part d'erreurs relatives > 0,25, consensus − raffiné, IC < 0 | IC inclut 0 ou > 0 |
| HP9 | le raffinement validé améliore les paramètres | banc : erreur relative raffiné − courbes, IC < 0 ; chaîne : C − `C_norefine` sur la part précise, IC > 0 | l'un des deux manque |
| HP10 | la naissance retrouve la primitive manquante | contexte exact, mesure indépendante de la polarité : nouvelle − ancienne, IC > 0, ET nouvelle ≥ 0,60 ; fausses naissances ≤ 0,10 | l'un des trois manque |
| HP11 | le rappel est limité par l'information, pas par l'algorithme | banc sans masque, 2 400 observations : part précise des courbes ≥ 0,85 pour chaque type ; chaîne dense : toutes présentes ≥ 0,25 à d4–6 | l'un des deux manque |
| HP12 | le coût reste compact | C ≤ 10 s par tâche (régimes primaires) et ≤ 3 × B | sinon |
| HP13 | C ne se dégrade pas hors distribution | régime `hard` : C − B sur l'accord, borne basse de l'IC > −0,01 | borne basse ≤ −0,01 |

HP1 à HP5 sont jugées sur les régimes primaires. Le seuil de 0,70 de HP1 traduit « saut qualitatif » (le niveau
actuel est d'environ 0,37 à 0,45 aux profondeurs 4–6) ; 0,9 reste l'ordre de grandeur visé, pas un seuil.

**Attentes déclarées** d'après la mise au point : HP1 à HP5 probablement non soutenues aux régimes primaires ; HP6
incertaine à 300 observations, probable à 2 400 ; HP10 probable ; HP11 incertaine (probable pour les droites, moins
pour les cercles et le secteur) ; HP12 probable.

### 0.7 Budget de calcul et critères d'arrêt

Un seul run AWS (`make deep-run-hp`, EC2 + SSM, `eu-north-1`, 2 × `c6i.xlarge`, 10 h au plus, terminaison
automatique). Estimation : 20 heures-CPU, soit environ 3 h de mur et moins de 2 USD. Arrêt : fin du run ; aucune
relance avec une configuration modifiée ; si plus de 5 % des sous-unités échouent, corriger le défaut et relancer
l'intégralité, en le signalant. Aucune itération sur les clés 1100–1131.

### 0.8 Règle de verdict, fixée d'avance

- `READY FOR 16-SEED CONFIRMATORY CAMPAIGN` : HP1, HP2, HP3, HP4, HP5 et HP12 toutes soutenues. Alors seulement,
  proposer la campagne — sans la lancer.
- `HIGH-PRECISION PRIMITIVES IMPROVED BUT INSUFFICIENT` : HP1 et HP2 soutenues (rappel ET précision progressent
  nettement aux régimes primaires), mais HP3, HP4 ou HP5 non soutenue.
- `HIGH-PRECISION PRIMITIVES STILL BLOCKING` : HP1 ou HP2 non soutenue.

Un gain limité au régime dense, au banc par primitive ou à la naissance contrôlée ne change pas le verdict ; il est
rapporté comme tel. Le lot scellé n'est lancé dans aucun cas.

## 1. Protocole exécuté

Run AWS `wce-deephp-20261010-092830` (EC2 + SSM, `eu-north-1`, 2 × `c6i.xlarge`) : 672 sous-unités par instance,
0 échec, 2 h 54 et 2 h 52, instances terminées. 74 512 lignes, 0 unité en erreur
(`results/deep_high_precision_summary.json`). 23 heures-CPU, mémoire résidente maximale 588 Mo, coût estimé 1,05 USD
(tarif approximatif). Aucun bras ajouté, aucune relance. 384 tâches aux régimes primaires, 192 au régime dense, 192 au
régime `hard`, 4 608 scènes de banc, 9 872 perturbations, 320 retraits d'atome. Une seule graine de tâches par clé :
rien ici n'est confirmatoire.

Écart au pré-enregistrement, à signaler : le bras `new_onebirth` (une seule naissance, sans raffinement ni
validation) figurait dans le code avant le lancement mais pas dans le tableau 0.4 ; il est rapporté comme
`POST-HOC / EXPLORATORY`.

## 2. Chaînes complètes, même aval fort

Régimes primaires (standard et rich, 384 tâches) :

| Chaîne | Accord | Final ≥ 0,99 | Final ≥ 0,97 | Candidate ≥ 0,97 | Rappel | Précis ≤ 0,10 | Fin ≤ 0,05 | Toutes présentes | Erreur param. | Hypothèses | Temps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (gelé) | 0,896 | 0,099 | 0,253 | 0,300 | 0,405 | 0,334 | 0,250 | 0,234 | 0,112 | 36,1 | 24,7 s |
| B (`flat_scene_k3`) | 0,892 | 0,096 | 0,266 | 0,255 | 0,433 | 0,321 | 0,180 | 0,208 | 0,059 | 14,1 | 2,5 s |
| B5 | 0,900 | 0,089 | 0,281 | 0,276 | 0,481 | 0,342 | 0,188 | 0,242 | 0,064 | 23,0 | 2,9 s |
| **C** | 0,912 | 0,076 | 0,305 | 0,331 | 0,567 | 0,419 | 0,273 | 0,307 | 0,060 | 44,4 | 4,6 s |
| `C_pool` (vivier seul) | 0,911 | 0,076 | 0,305 | 0,328 | 0,567 | 0,417 | 0,271 | 0,307 | 0,060 | 44,4 | 3,6 s |
| `C_prev` (sous-jalon précédent) | 0,907 | 0,076 | 0,281 | 0,339 | 0,540 | 0,405 | 0,240 | 0,287 | 0,061 | 46,6 | 8,1 s |
| `A_full` | 0,884 | 0,120 | 0,255 | 0,323 | — | — | — | — | — | — | 22,7 s |
| `B_full` | 0,903 | 0,146 | 0,346 | 0,406 | — | — | — | — | — | — | 18,0 s |

Différences appariées, régimes primaires :

| Différence | Accord | Final ≥ 0,99 | Candidate ≥ 0,97 | Rappel | Précis ≤ 0,10 | Toutes présentes |
|---|---|---|---|---|---|---|
| C − B | +0,019 [+0,013 ; +0,025] | −0,021 [−0,042 ; −0,003] | +0,075 [+0,042 ; +0,107] | +0,134 [+0,114 ; +0,155] | +0,098 [+0,079 ; +0,119] | +0,099 [+0,070 ; +0,130] |
| C − A | +0,015 [+0,009 ; +0,021] | −0,023 [−0,047 ; +0,000] | +0,031 [−0,003 ; +0,065] | +0,162 [+0,132 ; +0,192] | +0,086 [+0,060 ; +0,112] | +0,073 [+0,036 ; +0,112] |
| C − B5 | +0,012 [+0,007 ; +0,017] | −0,013 [−0,034 ; +0,005] | +0,055 [+0,023 ; +0,086] | +0,086 [+0,070 ; +0,102] | +0,077 [+0,060 ; +0,096] | +0,065 [+0,042 ; +0,091] |
| C − `C_prev` | +0,005 [−0,000 ; +0,010] | 0,000 | −0,008 [−0,039 ; +0,023] | +0,027 [+0,012 ; +0,042] | +0,014 [−0,002 ; +0,031] | +0,021 [−0,003 ; +0,044] |
| C − `B_full` | +0,009 [+0,004 ; +0,014] | −0,070 [−0,099 ; −0,044] | −0,075 [−0,109 ; −0,042] | — | — | — |
| C − `A_full` | +0,027 [+0,021 ; +0,034] | −0,044 [−0,070 ; −0,021] | +0,008 [−0,029 ; +0,044] | — | — | — |

Profondeurs 4–6 (192 tâches) : rappel de C 0,374 (B 0,229 ; A 0,160) ; précis 0,236 (B 0,146) ; toutes présentes
0,021 (B 0 ; A 0) ; accord 0,869 (B 0,840 ; A 0,852), C − B +0,030 [+0,019 ; +0,040] ; final ≥ 0,99 : 0 pour B et C.

D'où vient le gain de C. Entièrement des estimateurs spécialisés : C − `C_nocurves` accord +0,009 [+0,004 ; +0,014],
rappel +0,063, précis +0,060, toutes présentes +0,050. L'amélioration validée n'apporte rien aux régimes primaires :
`C_nobirth` est identique à C (aucune naissance retenue), C − `C_pool` accord +0,001 [−0,001 ; +0,002] ; un ensemble
amélioré n'est validé que dans 4,7 % des cas. La validation optimiste (`C_cv`) accepte beaucoup plus (2,4 ensembles
par tâche) et augmente la part précise du vivier (+0,017 [+0,011 ; +0,025]) sans effet sur l'accord.

Régime dense (2 400 observations, 192 tâches) :

| Chaîne | Accord | Final ≥ 0,99 | Final ≥ 0,97 | Candidate ≥ 0,97 | Rappel | Précis ≤ 0,10 | Fin ≤ 0,05 | Toutes présentes | Erreur param. | Temps |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0,938 | 0,260 | 0,484 | 0,526 | 0,482 | 0,447 | 0,390 | 0,240 | 0,079 | 47,3 s |
| B | 0,949 | 0,292 | 0,563 | 0,443 | 0,597 | 0,516 | 0,376 | 0,292 | 0,035 | 5,1 s |
| B5 | 0,952 | 0,271 | 0,568 | 0,505 | 0,653 | 0,551 | 0,387 | 0,349 | 0,041 | 5,5 s |
| **C** | 0,963 | 0,323 | 0,708 | 0,714 | 0,792 | 0,714 | 0,630 | 0,516 | 0,028 | 11,7 s |
| `C_pool` | 0,961 | 0,297 | 0,688 | 0,662 | 0,784 | 0,700 | 0,606 | 0,500 | 0,029 | 7,5 s |

Dense, appariées : C − B accord +0,014 [+0,008 ; +0,020], candidate +0,271 [+0,203 ; +0,333], rappel +0,195, précis
+0,199, toutes présentes +0,224 [+0,167 ; +0,286], final ≥ 0,97 +0,146 [+0,094 ; +0,203], final ≥ 0,99 +0,031
[−0,026 ; +0,088]. Profondeurs 4–6 : rappel 0,659 (B 0,412), toutes présentes 0,198 (B 0), candidate 0,479 (B 0,115),
final ≥ 0,99 0,115 (B 0,031), C − B +0,083 [+0,031 ; +0,146]. Au régime dense l'amélioration validée agit : C −
`C_pool` candidate +0,052 [+0,021 ; +0,088], final ≥ 0,99 à d4–6 +0,042 [+0,010 ; +0,083].

Hors distribution (`hard`, 192 tâches) : C − B accord −0,005 [−0,018 ; +0,008] ; accord en extrapolation −0,027
[−0,064 ; +0,010], et −0,060 [−0,116 ; −0,005] aux profondeurs 4–6 ; rappel +0,077 [+0,051 ; +0,105].

## 3. Rappel d'ensemble par nombre d'atomes

| Atomes | Rappel B | Toutes présentes B | Rappel C | Précis C | Toutes présentes C | `rappel^n` C | Toutes précises C | Rappel C dense | Toutes présentes C dense | `rappel^n` dense |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,969 | 0,969 | 0,984 | 0,953 | 0,984 | 0,984 | 0,953 | 1,000 | 1,000 | 1,000 |
| 2 | 0,539 | 0,219 | 0,742 | 0,516 | 0,563 | 0,551 | 0,281 | 0,922 | 0,875 | 0,850 |
| 3 | 0,401 | 0,063 | 0,552 | 0,339 | 0,234 | 0,168 | 0,047 | 0,854 | 0,625 | 0,623 |
| 4 | 0,297 | 0,000 | 0,465 | 0,313 | 0,047 | 0,047 | 0,000 | 0,734 | 0,313 | 0,291 |
| 5 | 0,250 | 0,000 | 0,381 | 0,250 | 0,016 | 0,008 | 0,000 | 0,681 | 0,281 | 0,147 |
| 6 | 0,141 | 0,000 | 0,276 | 0,146 | 0,000 | 0,000 | 0,000 | 0,563 | 0,000 | 0,032 |

Le rappel d'ensemble suit toujours `rappel^n`. À six atomes, il reste nul même au régime dense.

Par type (vivier, rappel / précis ≤ 0,10) :

| Type | B primaire | C primaire | B dense | C dense |
|---|---|---|---|---|
| demi-plan | 0,463 / 0,417 | 0,656 / 0,592 | 0,578 / 0,532 | 0,826 / 0,780 |
| bande | 0,342 / 0,296 | 0,451 / 0,366 | 0,620 / 0,563 | 0,739 / 0,725 |
| disque | 0,326 / 0,189 | 0,524 / 0,316 | 0,481 / 0,359 | 0,802 / 0,698 |
| anti-disque | 0,281 / 0,081 | 0,475 / 0,169 | 0,400 / 0,250 | 0,750 / 0,600 |
| anneau | 0,248 / 0,126 | 0,300 / 0,170 | 0,481 / 0,393 | 0,667 / 0,541 |
| secteur | 0,195 / 0,115 | 0,360 / 0,175 | 0,320 / 0,260 | 0,540 / 0,410 |

Par visibilité (rappel) : primaire C 0,120 / 0,316 / 0,685 (visibilité < 0,4 / 0,4–0,8 / > 0,8) contre B 0,058 / 0,203 /
0,487 ; dense C 0,248 / 0,684 / 0,927 contre B 0,058 / 0,368 / 0,751. Un atome bien visible est retrouvé 93 % du temps
au régime dense, 68 % aux régimes primaires.

## 4. Carte de sensibilité et précisions nécessaires

Un paramètre d'un atome perturbé, les autres exacts, sans finalisation. Part des tâches gardant final ≥ 0,99
(base : 91,6 % des tâches sont fonctionnelles sans perturbation) :

| Type | Famille | 0,02 | 0,05 | 0,10 | 0,20 | Tolérance |
|---|---|---|---|---|---|---|
| demi-plan | orientation | 0,98 | 0,38 | 0,21 | 0,12 | 0,02 |
| demi-plan | décalage | 0,97 | 0,80 | 0,30 | 0,14 | 0,05 |
| bande | orientation | 0,66 | 0,25 | 0,11 | 0,05 | **< 0,02** |
| bande | décalage | 0,96 | 0,34 | 0,14 | 0,05 | 0,02 |
| bande | largeur | 0,99 | 0,88 | 0,33 | 0,14 | 0,05 |
| disque | centre | 0,98 | 0,93 | 0,47 | 0,13 | 0,05 |
| disque | rayon | 0,96 | 0,66 | 0,18 | 0,05 | 0,02 |
| anti-disque | centre | 0,99 | 0,97 | 0,57 | 0,19 | 0,05 |
| anti-disque | rayon | 0,99 | 0,76 | 0,29 | 0,11 | 0,02 |
| anneau | centre | 0,99 | 0,57 | 0,15 | 0,01 | 0,02 |
| anneau | rayon (les deux) | 0,94 | 0,33 | 0,04 | 0,01 | 0,02 |
| anneau | largeur | 0,99 | 0,83 | 0,32 | 0,02 | 0,05 |
| secteur | sommet | 0,99 | 0,97 | 0,60 | 0,17 | 0,05 |
| secteur | direction | 0,92 | 0,40 | 0,17 | 0,03 | 0,02 |
| secteur | demi-ouverture | 0,93 | 0,38 | 0,19 | 0,05 | 0,02 |

Final ≥ 0,97 tolère beaucoup plus : presque toutes les familles tiennent à 0,05, et les centres à 0,10–0,20. Les
orientations et directions sont les paramètres critiques (un angle déplace toute la frontière, proportionnellement à
la distance) ; les centres et les largeurs sont les plus tolérants.

Erreur médiane des primitives du vivier, à comparer aux tolérances :

| Famille | B primaire | C primaire | C dense |
|---|---|---|---|
| demi-plan : orientation / décalage | 0,138 / 0,170 | 0,038 / 0,057 | 0,010 / 0,016 |
| bande : orientation / décalage / largeur | 0,142 / 0,321 / 0,237 | 0,040 / 0,175 / 0,207 | 0,010 / 0,020 / 0,026 |
| disque : centre / rayon | 0,537 / 0,165 | 0,210 / 0,084 | 0,032 / 0,013 |
| anti-disque : centre / rayon | 0,696 / 0,206 | 0,238 / 0,095 | 0,028 / 0,015 |
| anneau : centre / rayon | 0,661 / 0,399 | 0,471 / 0,329 | 0,039 / 0,031 |
| secteur : sommet / direction / demi-ouverture | 1,596 / 0,708 / 0,241 | 1,049 / 0,427 / 0,193 | 0,696 / 0,211 / 0,092 |

Part des vrais atomes ayant TOUS leurs paramètres dans la tolérance : primaire A 0,070, B 0,035, C 0,077 ; dense A
0,141, B 0,124, C 0,332. Par type pour C dense : demi-plan 0,66 ; disque 0,55 ; anti-disque 0,51 ; anneau 0,33 ;
secteur 0,08 ; bande 0,00 (sa tolérance d'orientation est sous la plus petite perturbation testée).

## 5. Banc par primitive (type connu, 4 608 scènes)

Part des scènes où la première hypothèse est précise (erreur relative ≤ 0,10) :

| Méthode | Toutes | Sans masque | Masque 0,3 | Masque 0,6 | 150 obs. | 300 | 600 | 2 400 | Catastrophiques (> 0,25) |
|---|---|---|---|---|---|---|---|---|---|
| gradient générique (cible) | 0,313 | 0,82 | 0,11 | 0,00 | 0,27 | 0,31 | 0,32 | 0,35 | 0,556 |
| gradient générique (contraste) | 0,336 | 0,56 | 0,34 | 0,11 | 0,23 | 0,33 | 0,36 | 0,43 | 0,574 |
| inférence existante | 0,427 | 0,77 | 0,37 | 0,15 | 0,24 | 0,41 | 0,49 | 0,58 | 0,390 |
| **courbes spécialisées** | 0,498 | 0,77 | 0,52 | 0,21 | 0,22 | 0,39 | 0,58 | 0,81 | 0,359 |
| recherche directe | 0,370 | 0,91 | 0,20 | 0,00 | 0,28 | 0,37 | 0,41 | 0,43 | 0,511 |
| courbes + raffinement | 0,404 | 0,87 | 0,28 | 0,06 | 0,27 | 0,40 | 0,45 | 0,50 | 0,440 |
| + consensus | 0,403 | 0,89 | 0,27 | 0,06 | 0,28 | 0,40 | 0,44 | 0,49 | 0,445 |
| raffinement depuis la vérité | 0,696 | 0,99 | 0,48 | 0,62 | 0,67 | 0,70 | 0,69 | 0,72 | 0,181 |

Bruit et masquage séparés (courbes / gradient / recherche directe) : ni l'un ni l'autre 0,81 / 0,84 / 0,94 ; bruit
seul 0,73 / 0,81 / 0,89 ; masquage seul 0,38 / 0,06 / 0,10 ; les deux 0,34 / 0,06 / 0,09. **Le masquage, pas le
bruit, est ce qui casse les estimateurs** ; les méthodes qui ajustent la primitive entière sur les étiquettes sont
les meilleures sans masque et s'effondrent dès 30 % de masquage.

Sans masque, par type et densité (courbes) : demi-plan 0,75 / 0,98 / 1,00 / 1,00 ; bande 0,39 / 0,88 / 1,00 / 1,00 ;
disque 0,47 / 0,83 / 0,97 / 1,00 ; anti-disque 0,27 / 0,45 / 0,83 / 1,00 ; anneau 0,08 / 0,44 / 0,84 / 1,00 ; secteur
0,53 / 0,83 / 0,97 / 1,00 (150 / 300 / 600 / 2 400 observations). Toutes les familles sont parfaitement résolues à
2 400 observations quand elles sont visibles ; à 300, l'anneau et l'anti-disque ne le sont qu'une fois sur deux.

Par type et masquage (courbes, masque 0 / 0,3 / 0,6) : demi-plan 0,93 / 0,80 / 0,50 ; bande 0,82 / 0,66 / 0,35 ; disque
0,82 / 0,52 / 0,19 ; anti-disque 0,64 / 0,39 / 0,10 ; anneau 0,59 / 0,38 / 0,09 ; secteur 0,83 / 0,36 / 0,01.

Appariées : courbes − gradient +0,185 [+0,170 ; +0,200] (positif pour chaque type, de +0,077 pour l'anneau à +0,378
pour le demi-plan) ; courbes − inférence existante +0,070 [+0,058 ; +0,083] (disque +0,175 ; bande −0,004, nul) ;
raffiné − courbes −0,094 [−0,106 ; −0,081] (le raffinement DÉGRADE) ; consensus − raffiné : catastrophiques +0,005
[+0,001 ; +0,009] ; recherche directe − courbes −0,127 ; contraintes partagées − courbes indépendantes +0,042
[+0,027 ; +0,057] (bande +0,055, anneau +0,030) ; vérité dans les cinq premiers − au premier rang : courbes +0,033
[+0,028 ; +0,038], inférence existante +0,057.

Oracles partiels (masque 0 / 0,3 / 0,6) : disque avec centre vrai 0,92 / 0,83 / 0,75 (sans : 0,82 / 0,52 / 0,19) ;
anti-disque 0,84 / 0,75 / 0,66 ; secteur avec sommet vrai 0,82 / 0,69 / 0,07 ; demi-plan avec orientation vraie
1,00 / 0,96 / 0,85 ; bande avec orientation vraie 0,85 / 0,77 / 0,61. Le paramètre limitant sous masquage est le
**centre** pour les primitives circulaires, le **sommet** pour le secteur, l'**orientation** pour les primitives
rectilignes.

Identifiabilité (diagnostic `PrimitiveIdentifiability` sur la vraie primitive, sans masque) : identifiable dans
0,87 (demi-plan), 0,90 (bande), 0,77 (disque), 0,72 (anti-disque), 0,90 (anneau), 0,68 (secteur) des scènes. Quand
la primitive est dite identifiable, les courbes sont précises dans 0,98 / 0,87 / 0,92 / 0,78 / 0,65 / 0,91 des cas ;
quand elle est dite non identifiable, 0,63 / 0,34 / 0,21 / 0,24 / 0,08 / 0,11. Sous masquage ce diagnostic n'est PAS
valide : il raisonne sur la primitive seule, sans le masque, et déclare presque tout non identifiable.

## 6. Naissance d'une primitive manquante

Un vrai atome retiré, les autres exacts (320 tâches) :

| Bras | Né | Retrouvé (polarité libre) | Retrouvé ≤ 0,10 | Final ≥ 0,99 après | Accord après | Temps |
|---|---|---|---|---|---|---|
| avant | — | — | — | 0,031 | 0,822 | — |
| ancienne recherche sur résidu | 16,7 hypothèses | 0,459 | 0,247 | 0,178 | 0,902 | 3,4 s |
| nouvelle, validée (bras pré-enregistré) | 0,79 | 0,484 | 0,328 | 0,172 | 0,942 | 2,5 s |
| nouvelle sans morphologie | 0,78 | 0,466 | 0,316 | 0,181 | 0,942 | 2,2 s |
| nouvelle non validée | 0,97 | 0,556 | 0,356 | 0,175 | 0,947 | 0,9 s |
| une seule naissance, ni raffinement ni validation (`POST-HOC`) | 0,95 | 0,719 | 0,519 | 0,522 | 0,973 | 0,6 s |

Appariées contre l'ancienne : nouvelle validée, polarité libre +0,025 [−0,038 ; +0,084], précise +0,081
[+0,025 ; +0,138], accord +0,040 [+0,018 ; +0,065] ; non validée +0,097 [+0,034 ; +0,156] ; une seule naissance
+0,259 [+0,206 ; +0,316], final ≥ 0,99 +0,344 [+0,288 ; +0,400]. Morphologie : nouvelle − sans morphologie +0,019
[−0,009 ; +0,047], nul ; le classifieur répond « arc » pour presque tous les types manquants (65 bandes sur 84, 41
demi-plans sur 52) : il n'est pas informatif.

Par profondeur (nouvelle validée, polarité libre) : 0,81 / 0,50 / 0,48 / 0,34 / 0,28 de deux à six atomes. Contexte
perturbé de 0,05 : nouvelle 0,447, ancienne 0,394, et final ≥ 0,99 presque nul pour tous (0,04 au mieux) : la
naissance ne compense pas des voisins imprécis. Fausses naissances sur 384 ensembles complets : **0**. Erreurs de
paramètres des primitives nées et retrouvées : centre 0,03–0,05, rayon 0,02, orientation 0,015–0,03, décalage
0,03–0,04 ; secteur : sommet 0,23.

**Anti-disque, le `0/42` du sous-jalon précédent : c'était un défaut de MESURE.** Ici encore, l'ancienne mesure
(accord fonctionnel sans liberté de polarité, après retrait de la négation) donne 0 sur 24 pour l'ancienne méthode
comme pour la nouvelle. Avec la polarité libre : 0,542 (ancienne) et 0,417 (nouvelle validée), 0,542 non validée.
Le disque de même frontière est bien trouvé ; il est fusionné avec son complément dans une seule hypothèse
représentée par le disque, et l'ancienne mesure comparait un anti-disque à un disque. Aucune des six causes
envisagées (frontière, objectif, masque, initialisation, score) n'est en jeu ; seule la polarité de la mesure
l'était. Le final ≥ 0,99 après naissance d'un anti-disque (0,42–0,46) est d'ailleurs le plus élevé de tous les types.

## 7. Hypothèses pré-enregistrées

| # | Résultat | Statut |
|---|---|---|
| HP1 | C − B +0,134, C − A +0,162, IC > 0 ; mais rappel à d4–6 = 0,374 (seuil 0,70 ; réfutation < 0,60) | **réfutée** (gain réel, saut qualitatif absent) |
| HP2 | précis : C − B +0,098 [+0,079 ; +0,119] ; C − A +0,086 [+0,060 ; +0,112] | soutenue |
| HP3 | toutes présentes : C − B +0,099, C − A +0,073, IC > 0 ; mais 0,021 à d4–6 (seuil 0,25 ; réfutation < 0,10) | **réfutée** |
| HP4 | candidate : C − B +0,075, IC > 0 ; C − A +0,031 [−0,003 ; +0,065] | **non soutenue** |
| HP5 | accord : IC > 0 contre A et B ; final ≥ 0,99 : C − B −0,021 [−0,042 ; −0,003] | **réfutée** |
| HP6 | courbes − gradient +0,185 [+0,170 ; +0,200], positif pour chaque type | soutenue |
| HP7 | banc : +0,033 (< 0,05) ; chaîne : C − `C_k1` accord +0,003 [−0,002 ; +0,008] | **réfutée** |
| HP8 | catastrophiques, consensus − raffiné +0,005 [+0,001 ; +0,009] | **réfutée** |
| HP9 | banc : raffiné − courbes, erreur +0,029 [+0,019 ; +0,038] ; chaîne : +0,002 [0 ; +0,005] | **réfutée** |
| HP10 | nouvelle − ancienne +0,025 [−0,038 ; +0,084] ; nouvelle 0,484 (< 0,60) ; fausses naissances 0 | **réfutée** |
| HP11 | banc sans masque à 2 400 observations : 1,00 pour chaque type ; chaîne dense, toutes présentes à d4–6 : 0,198 (< 0,25) | **non soutenue** (première moitié seule) |
| HP12 | C 4,6 s (≤ 10 s) ; 1,9 × B (≤ 3 ×) | soutenue |
| HP13 | `hard` : C − B accord −0,005, borne basse −0,018 (≤ −0,01) | **réfutée** |

Attentes déclarées : HP1 à HP5 probablement non soutenues — confirmé, sauf HP2, soutenue ; HP6 confirmée ; HP10
démentie ; HP11 à moitié ; HP12 confirmée.

## 8. Réponses aux dix questions

**1. Quel type de primitive est le principal responsable du rappel insuffisant ?** Aucun seul : tous sont sous 0,70
aux régimes primaires. Les plus faibles sont l'anneau (0,30) et le secteur (0,36), puis la bande (0,45) et
l'anti-disque (0,48) ; le demi-plan est le meilleur (0,66). En précision, anneau, anti-disque et secteur sont à
0,17. Les primitives à deux courbes (anneau, bande, secteur) paient deux fois l'extraction.

**2. Quels paramètres sont les plus difficiles ?** À estimer : le sommet du secteur (erreur médiane 1,05, encore
0,70 au régime dense), le centre des primitives circulaires (0,21 à 0,47), puis les rayons de l'anneau. À tolérer :
les orientations et directions, qui cassent la composition dès 0,05. Sous masquage, l'oracle partiel montre que le
centre (cercles), le sommet (secteur) et l'orientation (droites) sont le paramètre limitant.

**3. Quelles précisions sont nécessaires par famille ?** Pour garder final ≥ 0,99 : orientations, directions,
demi-ouverture, rayons, décalage d'une bande : 0,02 ; centres de disque et d'anti-disque, sommet de secteur,
largeurs, décalage d'un demi-plan : 0,05 ; orientation d'une bande : moins de 0,02 ; centre d'un anneau : 0,02.
Pour final ≥ 0,97, environ 0,05 partout et 0,10 à 0,20 pour les centres. Les erreurs médianes de C aux régimes
primaires sont deux à vingt fois au-dessus ; au régime dense elles passent sous la tolérance pour les droites et
les cercles, pas pour le secteur.

**4. Les estimateurs spécialisés battent-ils le gradient générique ?** Oui en moyenne (+0,185) et pour chaque type,
et ils battent l'inférence existante (+0,070). Mais le tableau se renverse selon la condition : sans masque, le
gradient (0,82) et la recherche directe (0,91) font mieux que les courbes (0,77) ; dès 30 % de masquage ils tombent
à 0,11–0,20 quand les courbes tiennent à 0,52. En composition, tout est masqué : ce sont les courbes qui comptent.

**5. Top-k paramétrique aide-t-il ?** Peu. Banc : +0,033 de vérité entre le premier et le cinquième rang. Chaîne :
`k = 5` contre `k = 1`, rappel +0,050 et toutes présentes +0,034, mais accord +0,003 non significatif.

**6. Le consensus améliore-t-il la robustesse ?** Le consensus SÉQUENTIEL de l'extraction, oui : c'est lui qui porte
le gain des courbes. La médiane de rééchantillonnages, non : elle ajoute 0,5 point d'erreurs catastrophiques.

**7. La recherche sur résidu récupère-t-elle réellement les primitives manquantes ?** Partiellement, et pas mieux
qu'avant dans sa forme pré-enregistrée (0,48 contre 0,46, non significatif), bien que plus précise (+0,081) et sans
aucune fausse naissance. La forme la plus simple — une naissance, sans validation ni raffinement — retrouve 0,72 et
porte le final ≥ 0,99 de 0,03 à 0,52 ; ce résultat est post-hoc et à confirmer. Dans la chaîne réelle, aucune
naissance n'est retenue : la validation hors-échantillon est trop sévère à 300–600 observations, et le contexte y
est imprécis.

**8. Le secteur et l'anti-disque sont-ils mieux compris ?** L'anti-disque, oui : son échec total était un défaut de
mesure de polarité, et il se comporte comme un disque (un peu moins bien retrouvé, 0,48 contre 0,52). Le secteur,
partiellement : parfaitement résolu quand il est visible et dense (1,00 à 2 400 observations), il s'effondre sous
masquage (0,36 puis 0,01) parce que le sommet n'est connu que par l'intersection de deux demi-droites courtes ;
avec le sommet vrai il remonte à 0,69 à 30 % de masquage. Il reste le type le moins bien retrouvé au régime dense
(0,54) et le seul dont les paramètres y restent hors tolérance.

**9. Le rappel individuel approche-t-il un niveau compatible avec six atomes ?** Non aux régimes primaires : 0,57 en
moyenne, 0,37 aux profondeurs 4–6, 0,28 à six atomes. Au régime dense : 0,79 en moyenne, 0,66 aux profondeurs 4–6,
0,56 à six atomes, et 0,93 pour les atomes bien visibles. L'ordre de grandeur de 0,9 n'est atteint que pour les
atomes visibles, avec huit fois plus d'observations.

**10. Le gain traverse-t-il réellement jusqu'à la composition finale ?** En partie seulement. Aux régimes primaires,
l'accord monte (+0,019 contre B, +0,015 contre A) et la structure candidate aussi contre B, mais final ≥ 0,99
baisse (−0,021). Au régime dense, la structure candidate ≥ 0,97 passe de 0,44 à 0,71 et final ≥ 0,97 de 0,56 à
0,71, mais final ≥ 0,99 seulement de 0,29 à 0,32 (non significatif ; +0,083 aux profondeurs 4–6). La bonne
structure est souvent dans les candidates sans être celle qui est choisie ou finalisée.

## 9. Limites

- Partition dev, une graine de tâches par clé ; 32 tâches par profondeur au régime dense.
- HP1 et HP3 sont réfutées sur leur seuil de niveau, pas sur l'existence d'un gain : les quatre IC sont au-dessus de
  zéro. La règle écrite d'avance est appliquée telle quelle.
- Le régime dense n'a pas servi à la mise au point de la chaîne, mais le diagnostic de mise au point à 3 000
  observations a motivé l'extracteur ; ses résultats sont secondaires.
- Tolérances mesurées avec UN paramètre perturbé et les autres atomes exacts : elles bornent le besoin par le bas ;
  plusieurs erreurs simultanées se cumulent. La plus petite perturbation testée est 0,02.
- « Précis ≤ 0,10 » est une erreur relative à l'aire de la primitive, plus sévère que le 0,95 d'accord fonctionnel
  du sous-jalon précédent ; les deux sont rapportées.
- Le diagnostic d'identifiabilité n'est valide que sans masque.
- Le raffinement et le consensus du banc ajustent la primitive SEULE sur les étiquettes : sous masquage ce critère
  est faux par construction, ce qui explique une part de leur dégradation ; même le raffinement depuis la vérité y
  tombe à 0,48–0,62.
- Le classifieur de morphologie du résidu n'est pas informatif ; la « structure du résidu » n'est donc pas exploitée
  comme prévu.
- Le bras « une seule naissance » est post-hoc.
- Tarifs AWS approximatifs.
- Aucun composant appris testé.

## 10. Nouveau verrou

Trois constats se dégagent, dans l'ordre d'importance.

1. **Le rappel par atome est limité par la quantité d'information par frontière visible, et par le masquage.** À
   2 400 observations et sans masque, chaque famille est résolue à 100 % ; à 300 observations ou à 60 % de
   masquage, aucune ne l'est. Les régimes primaires (300 et 600 observations pour jusqu'à six atomes) sont en deçà
   de ce que les estimateurs géométriques peuvent résoudre.
2. **Même quand les atomes sont là, final ≥ 0,99 ne suit pas.** Au régime dense, la moitié des tâches ont toutes
   leurs primitives et 71 % ont une structure candidate ≥ 0,97, mais 32 % seulement finissent à 0,99 : les
   tolérances de 0,02 sur les angles et les rayons ne sont tenues que par un tiers des atomes. Le verrou se déplace
   vers la **précision finale conjointe** et la **sélection** de la bonne structure parmi les candidates.
3. **La validation hors-échantillon neutralise l'amélioration aux régimes primaires** (4,7 % d'acceptation), alors
   qu'une naissance non validée fait passer final ≥ 0,99 de 0,03 à 0,52 en contexte exact. Le compromis
   puissance / sûreté de la validation est mal réglé pour 300 observations.

## 11. Verdict

`HIGH-PRECISION PRIMITIVES STILL BLOCKING`

Règle pré-enregistrée : HP1 non soutenue (rappel de 0,37 aux profondeurs 4–6, pour un seuil de 0,70). Le gain est
réel et significatif sur le rappel, la précision, le rappel d'ensemble et l'accord, contre le pipeline gelé comme
contre `flat_scene_k3`, mais ce n'est pas le saut qualitatif visé, et final ≥ 0,99 recule aux régimes primaires.
Aucune campagne 16 graines proposée ; lot scellé intact ; rien branché par défaut ; `flat_scene_k3` reste *confirmed
dev baseline candidate*.

# Rapport — WCE Validated Optimizer

Sous-jalon EXPLORATOIRE du jalon « Deep Compositional Structure Discovery & Model-Class Boundary ».
Données : partition **dev uniquement**, **vrais atomes fournis**. Ni la validation, ni le test confirmatoire, ni le lot
scellé n'ont été touchés. Tout est `POST-HOC / EXPLORATORY`.

Le diagnostic qui motive ce sous-jalon (`docs/rapport_downstream_selection_loss.md`) a été commité AVANT la correction
(commit `3074abe`) et n'a pas été réécrit.

Sources : `results/deep_validated_summary.json` ; runs AWS `wce-deepvo-20261004-162019` (expérience à deux bras, 768
sous-unités) et `wce-deepopt-20261004-163055` (diagnostic de l'optimiseur, 64 tâches), aucune unité en erreur.
Cibles : `make deep-run-arms`, `make deep-run-optimizer`, `make deep-vo-aggregate`, `make deep-vo-report`.

Précision sur les effectifs : chaque jeu de « 128 tâches » contient 64 structures distinctes, chacune observée dans
deux régimes (standard et riche). Les deux jeux n'ont aucune structure en commun.

## Verdict

| Question | Réponse |
|---|---|
| Mauvais optimiseur (mécanisme) ? | Secondaire. Le pas d'Adam éjecte un bon point en 5 pas, mais réduire le pas ou la douceur ne change pas le point d'arrivée. |
| Mauvais objectif de substitution ? | **Oui, cause principale.** L'objectif continu s'améliore dans 94 % des cas pendant que l'accord réel se dégrade dans 100 %. |
| Instabilité numérique ? | Présente, non causale. Les deux ajusteurs et les deux précisions aboutissent à la même issue. |
| Mauvais critère d'acceptation ? | **Oui, second facteur.** Sans critère : 7 % de récupération ; critère de substitution : 16 % ; vraisemblance profilée : 54 %. |

Le garde-fou généralise et le critère fixé d'avance est atteint. Il ne récupère pas tout : ne pas ajuster du tout ferait
mieux sur ces tâches où les paramètres de départ sont déjà justes.

## La correction (un seul étage)

`core/deep_fit.py` : `validated_fit` — état initial → proposition de l'optimiseur → évaluation → commit ou rollback.

- **Périmètre** : uniquement l'ajustement d'un candidat EXISTANT (une initialisation, qui est l'état du candidat). Les
  ajustements à plusieurs initialisations aléatoires (atomes) n'ont pas d'état initial à protéger et ne sont pas
  concernés. Générateur, simplification, classement, MDL, métrique, découverte de primitives et politique de mesure sont
  inchangés ; l'optimiseur lui-même est inchangé.
- **Critère** : vraisemblance profilée des prédictions dures sur les observations — celle que le système utilise ensuite
  pour enregistrer et classer. Aucune vérité.
- **Tolérance** : 1e-9 nat.
- **Égalité** (écart ≤ tolérance) : rollback — sans preuve d'amélioration, l'état initial est conservé.
- **NaN / inf** dans les paramètres proposés ou dans leur critère : rollback. Si le critère de l'état initial est non fini
  et la proposition finie : commit.
- **Rollback** : l'état initial est un tenseur distinct, jamais modifié en place (vérifié par une assertion) ; revenir en
  arrière consiste à le rendre tel quel, pas à le reconstruire.
- Le budget est débité même en cas de rollback (le calcul a eu lieu).

## 1. Le garde-fou généralise-t-il sur les 128 nouvelles tâches ?

Oui. Critère fixé avant résultat : au moins +20 points de récupération au seuil 0,99, sans baisse d'accord moyen.

| | Bras A (pipeline gelé) | Bras B (garde-fou) | Différence appariée | IC 95 % | Taille d'effet |
|---|---|---|---|---|---|
| Récupération ≥ 0,99 | 7,0 % | 53,9 % | **+46,9 points** | [+38,2 ; +55,6] | 0,94 |
| Accord fonctionnel moyen | 0,958 | 0,972 | +0,013 | [+0,010 ; +0,017] | 0,61 |

Critère atteint. Sur les 128 tâches du diagnostic : 9,4 % → 59,4 % (+50,0 points, IC [41,3 ; 58,7]), très proche du
contrefactuel annoncé (57 %).

## 2. Combien de récupération est réellement restaurée ?

Nouvelles tâches :

| Seuil | Bras A | Bras B | Différence | IC 95 % |
|---|---|---|---|---|
| ≥ 0,90 | 94,5 % | 94,5 % | 0,0 | [−2,2 ; +2,2] |
| ≥ 0,95 | 79,7 % | 82,0 % | +2,3 | [−1,7 ; +6,4] |
| ≥ 0,97 | 53,9 % | 69,5 % | +15,6 | [+9,3 ; +21,9] |
| ≥ 0,99 | 7,0 % | 53,9 % | +46,9 | [+38,2 ; +55,6] |

| Nouvelles tâches | Bras A | Bras B |
|---|---|---|
| Vraisemblance profilée de la solution retenue | −131,8 | −129,0 |
| Erreurs sur les observations | 39,6 | 38,4 |
| Accord en zone observée | 0,959 | 0,975 |
| Structure exacte / équivalente | 39 % / 51 % | 45 % / 57 % |
| Nombre d'atomes retenus | 3,05 | 3,05 |
| Ajustements acceptés / annulés par tâche | — | 6,1 / 5,3 |
| Équivalents-ajustement | 28,1 | 27,2 |

Le gain est concentré au sommet de l'échelle : le garde-fou ne change rien aux seuils 0,90 et 0,95, il transforme des
solutions « presque bonnes » en solutions exactes. Tableau croisé au seuil 0,99 : 60 tâches récupérées par B seul, 9 par
les deux, 59 par aucun, **0 par A seul**. Près de la moitié des ajustements (46 %) sont annulés.

Plafond : sans aucun ajustement, le candidat classé premier avant ajustement atteint 0,99 dans 86 à 89 % des tâches. Le
garde-fou restaure donc environ 60 % de ce qui est restaurable.

## 3. Dégrade-t-il certaines tâches ?

Oui, rarement. Sur les nouvelles tâches : accord amélioré dans 59,4 % des tâches, inchangé dans 35,9 %, **dégradé dans
4,7 %** (6 tâches). Sur les tâches du diagnostic : 61,7 % / 35,9 % / 2,3 %.

Pires cas : 0,964 → 0,854 et 0,900 → 0,872 (nouvelles tâches, standard, profondeur 4) ; 0,951 → 0,841 (diagnostic,
standard, profondeur 4). Aucune de ces tâches n'était au-dessus de 0,99 dans le bras A : le garde-fou n'a fait perdre
aucune récupération. Le mécanisme n'a pas été analysé en détail ; l'explication la plus probable est que conserver un
candidat différent change les atomes réinjectés aux tours suivants, et donc la structure finalement classée première.

## 4. Le bénéfice est-il stable par profondeur et régime ?

Par profondeur, oui. Par régime, non.

Nouvelles tâches, récupération ≥ 0,99 :

| | Bras A | Bras B | Différence | IC 95 % | Δ accord moyen | Améliorées / dégradées |
|---|---|---|---|---|---|---|
| d2 | 25,0 % | 81,3 % | +56,3 | [+38,4 ; +74,1] | +0,013 | 78 % / 0 % |
| d3 | 0,0 % | 62,5 % | +62,5 | [+45,1 ; +79,9] | +0,018 | 63 % / 0 % |
| d4 | 3,1 % | 40,6 % | +37,5 | [+20,1 ; +54,9] | +0,007 | 47 % / 13 % |
| d5 | 0,0 % | 31,3 % | +31,3 | [+14,6 ; +47,9] | +0,016 | 50 % / 6 % |
| Standard | 1,6 % | 37,5 % | +35,9 | [+24,0 ; +47,9] | +0,011 | 47 % / 8 % |
| Riche | 12,5 % | 70,3 % | +57,8 | [+45,5 ; +70,1] | +0,016 | 72 % / 2 % |

Par cellule (16 tâches chacune) :

| Nouvelles tâches | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Standard, A → B | 6 % → 75 % | 0 % → 56 % | 0 % → 13 % | 0 % → 6 % |
| Riche, A → B | 44 % → 88 % | 0 % → 69 % | 6 % → 69 % | 0 % → 56 % |

| Tâches du diagnostic | d2 | d3 | d4 | d5 |
|---|---|---|---|---|
| Standard, A → B | 6 % → 56 % | 0 % → 44 % | 0 % → 31 % | 0 % → 13 % |
| Riche, A → B | 50 % → 81 % | 19 % → 75 % | 0 % → 88 % | 0 % → 88 % |

Le garde-fou préserve aussi les structures profondes en régime riche (56 à 88 % à la profondeur 5). En régime standard,
le bénéfice s'éteint aux profondeurs 4 et 5 sur les nouvelles tâches : +12,5 points [−5,7 ; +30,7] et +6,3 points
[−7,1 ; +19,6], intervalles incluant zéro, et un accord moyen qui ne progresse pas à la profondeur 4 (−0,003
[−0,021 ; +0,016]). C'est là aussi que se trouvent les dégradations.

## 5. Pourquoi l'optimiseur déplace-t-il des solutions déjà bonnes ?

Diagnostic sur 64 tâches, deux points de départ quasi parfaits : la vraie structure aux vrais paramètres (accord 1,000)
et le candidat classé premier avant ajustement (accord 0,996). L'optimiseur n'a pas été modifié ; les variantes
ci-dessous sont des mesures. La trace reproduit `fit_batch` exactement (écart nul).

### Trajectoire moyenne depuis les vrais paramètres

| Pas | Raideur | Accord réel | Erreurs sur les données | Objectif continu (vraisemblance robuste) | Vraisemblance profilée | Norme du gradient | Déplacement max |
|---|---|---|---|---|---|---|---|
| 0 | 4 | 1,000 | 37,9 | −137,0 | −128,7 | 40,1 | 0,00 |
| 5 | 5 | 0,926 | 61,3 | −165,7 | −176,8 | 25,0 | 0,22 |
| 10 | 6 | 0,937 | 54,4 | −154,4 | −163,4 | 19,8 | 0,33 |
| 20 | 8 | 0,952 | 47,8 | −142,5 | −150,3 | 17,0 | 0,39 |
| 40 | 12 | 0,960 | 43,2 | −136,5 | −140,6 | 9,9 | 0,47 |
| 60 | 16 | 0,963 | 41,4 | −134,9 | −136,7 | 6,7 | 0,52 |
| 80 | 20 | 0,963 | 40,3 | −134,0 | −134,1 | 11,5 | 0,59 |
| 100 | 24 | 0,964 | 39,8 | −133,8 | −132,8 | — | 0,61 |

La solution se dégrade **dans les 5 premiers pas, dans 100 % des tâches**. Les vrais paramètres ne sont pas un point
stationnaire de l'objectif continu (norme du gradient 40 à la raideur 4, 80 à la raideur 24) ; or Adam avance d'environ
un pas d'apprentissage par itération quelle que soit la taille du gradient : en 5 pas, chaque paramètre a bougé
d'environ 0,22. L'optimiseur remonte ensuite vers un optimum de l'objectif continu qui est **meilleur que le point de
départ pour cet objectif** (−133,8 contre −137,0) mais **moins bon pour le monde** (accord 0,964) et moins bon sur les
données discrètes (39,8 erreurs contre 37,9).

### Hypothèses A à J

| Hypothèse | Mesure | Conclusion |
|---|---|---|
| A. Frontière initialement trop adoucie | raideur constante 24 : accord final 0,973 (contre 0,964), récupération 12,5 % (contre 12,5 %) | effet faible ; n'explique pas la perte |
| B. Pas d'apprentissage trop élevé | pas 0,02 : 0,969 ; pas 0,005 : 0,974 ; récupération 12,5 % | effet faible ; réduit le déplacement (0,61 → 0,16), pas la dégradation |
| A + B | raideur 24 et pas 0,005 : accord 0,975, récupération 12,5 % | les deux réglages ensemble ne restaurent rien |
| C. Mauvais conditionnement | conditionnement médian de la hessienne à la raideur 24 : 151 ; supérieur à 10⁴ dans 9 % des tâches | modéré ; pas la cause générale |
| D. Gradient aligné sur une approximation | objectif continu amélioré dans 94 % des cas, accord dégradé dans 100 % ; gradient non nul aux vrais paramètres | **soutenue** |
| E. Saturation | 18 % des points saturés à la raideur 4, 75 % à la raideur 24 | mesurée ; pas d'effet propre isolé |
| F. Paramètres non identifiables | direction plate dans 8 à 13 % des tâches ; courbure négative aux vrais paramètres dans 48 % | contribue : la vérité n'est pas un minimum local de l'objectif dans la moitié des tâches |
| G. Interaction entre atomes | 1,5 à 3,4 % de la masse de la hessienne est hors des blocs par atome | faible ; rejetée |
| H. Mauvais calendrier de raideur | calendrier inversé : 0,936 ; constant ≈ croissant | le calendrier actuel n'est pas le problème |
| I. Optimum continu ≠ optimum fonctionnel | meilleur état selon l'objectif continu : accord 0,968 ; l'objectif monte et la vraisemblance profilée baisse dans 58 % des cas | **soutenue — cause principale** |
| J. Instabilité numérique | simple contre double précision : accord différent de plus de 0,002 dans 1,6 % des tâches (maximum 0,004) | présente, négligeable |

Toutes les variantes de l'optimiseur dégradent l'accord dans 100 % des tâches, sauf la descente monotone (section 8).

## 6. L'objectif continu est-il mal aligné avec la qualité discrète ?

Oui. Depuis les vrais paramètres, configuration par défaut :

| | |
|---|---|
| L'objectif continu s'améliore | 94 % des tâches |
| L'accord réel se dégrade | 100 % |
| L'objectif continu s'améliore ET l'accord se dégrade | 94 % |
| L'objectif continu s'améliore ET la vraisemblance profilée (discrète) se dégrade | 58 % |
| La vraisemblance profilée s'améliore | 23 % |
| Corrélation de rang, le long de la trajectoire, entre objectif continu et accord | 0,51 en moyenne ; négative dans 16 % des tâches |
| … entre vraisemblance profilée et accord | 0,57 |
| … entre la perte optimisée (à raideur courante) et accord | 0,37 |

La corrélation moyenne est positive parce que les deux quantités chutent ensemble au début puis remontent ensemble ; ce
qui diverge, c'est le point d'arrivée. C'est un problème d'objectif de substitution : l'optimiseur fait ce qu'on lui
demande, et ce qu'on lui demande n'est pas la qualité fonctionnelle.

Le critère discret lui-même n'est pas parfaitement aligné : quand la vraisemblance profilée s'améliore (23 % des cas),
l'accord baisse quand même. Avec 300 à 600 observations bruitées, un état peut mieux expliquer les données que la vérité
tout en étant fonctionnellement moins bon. C'est ce qui limite le garde-fou.

## 7. L'ancien et le nouveau fitter ont-ils des comportements scientifiquement différents ?

Numériquement oui, scientifiquement non.

| Même point de départ, 100 pas | Vrais paramètres (64 tâches) | Premier candidat (64 tâches) | Candidats du diagnostic aval (128 tâches) |
|---|---|---|---|
| Écart de paramètres > 1e-6 | 33 % | 36 % | — |
| Écart de paramètres > 1e-3 | 4,7 % | 1,6 % | 28 % |
| Écart maximal de paramètres | 0,13 | 0,0014 | 1,01 |
| Écart maximal d'accord | 0,0075 | 0,0005 | 0,035 |
| Accord moyen, ancien / nouveau | 0,9640 / 0,9639 | 0,9629 / 0,9629 | — |
| Erreurs sur les données, ancien / nouveau | 39,66 / 39,75 | 40,31 / 40,31 | — |
| Même issue au seuil 0,99 | 100 % | 100 % | — |
| Même premier classé / même ordre complet | — | — | 98,4 % / 86,7 % |
| Temps par ajustement isolé, ancien / nouveau | 0,200 s / 0,181 s | 0,192 s / 0,176 s | — |

L'origine de l'écart est l'arrondi de l'initialisation en simple précision dans l'ancien ajusteur (≈ 1e-7), amplifié par
100 pas d'optimisation. Les deux ajusteurs dégradent autant et aboutissent à la même issue.

**H10(b) (écart ≤ 1e-6) est réfutée sur les composites.** L'ajusteur par lots n'est plus qualifié de numériquement
équivalent : c'est une autre implémentation, avec un compromis mesuré — environ 7 fois plus rapide par lot de 8
initialisations, 1,1 fois pour un ajustement isolé, même premier classé dans 98 % des tâches, même ordre complet dans
87 %. La documentation de `core/deep_fit.py` a été corrigée en ce sens.

## 8. Un best-so-far ou un early stopping suffit-il ?

Non, pas mieux que de ne pas optimiser.

| Départ aux vrais paramètres | Accord moyen | Récupération ≥ 0,99 |
|---|---|---|
| Sans optimiser (baseline « no-op ») | 1,000 | 100 % |
| État final de l'optimiseur | 0,964 | 12,5 % |
| Meilleur état rencontré selon l'objectif continu | 0,968 | 19 % |
| Meilleur état rencontré selon la vraisemblance profilée | 0,988 | 70 % |
| Arrêt précoce (dès que la vraisemblance profilée passe sous sa valeur initiale) | 1,000 | 100 % |
| Descente monotone à raideur fixe | 0,998 | 94 % |

| Départ au premier candidat | Accord moyen | Récupération ≥ 0,99 |
|---|---|---|
| Sans optimiser | 0,996 | 86 % |
| État final | 0,963 | 11 % |
| Meilleur état selon l'objectif continu | 0,967 | 17 % |
| Meilleur état selon la vraisemblance profilée | 0,987 | 63 % |
| Arrêt précoce | 0,996 | 86 % |
| Descente monotone | 0,994 | 80 % |

- **Bénéfice de l'optimisation : nul. Dommage : −0,036 d'accord et −75 à −88 points de récupération.** Sur ces points de
  départ, l'optimisation n'apporte rien et détruit.
- L'arrêt précoce s'arrête au pas 0 dans toutes les tâches : il équivaut à ne pas optimiser.
- Le meilleur état selon la vraisemblance profilée est l'état initial dans 64 à 67 % des cas ; sinon c'est un état un peu
  plus vraisemblable et fonctionnellement moins bon.
- **Monotonie** : l'optimiseur peut être rendu monotone par rapport à l'objectif qu'il prétend optimiser (un pas n'est
  gardé que s'il l'améliore). Mais il ne bouge alors presque plus (déplacement 0,009) : la descente monotone « réussit »
  en étant un quasi no-op. Et si elle avançait, elle irait vers l'optimum de l'objectif continu, qui est le mauvais
  endroit. Forcer la monotonie sur un critère mal aligné ne résout rien.

## 9. Le vrai problème est-il le mécanisme d'optimisation ou l'objectif optimisé ?

L'objectif.

- Mécanisme : changer le pas, la raideur ou le calendrier ne change pas la récupération (12,5 % dans toutes les
  variantes). Le mécanisme explique la brutalité de la chute initiale, pas le point d'arrivée.
- Objectif : l'optimum de l'objectif continu est à un accord de 0,964 à 0,975, quelle que soit la façon d'y aller.
- Critère d'acceptation, mesuré par le bras C (même garde-fou, mais critère = objectif continu) :

| Nouvelles tâches | Récupération ≥ 0,99 | Accord moyen | Ajustements acceptés / annulés |
|---|---|---|---|
| A — aucun critère | 7,0 % | 0,958 | — |
| C — critère de substitution | 16,4 % | 0,962 | 11,6 / 0,6 |
| B — vraisemblance profilée | 53,9 % | 0,972 | 6,1 / 5,3 |

Le critère de substitution accepte presque tout (95 % des propositions), puisque c'est ce que l'optimiseur améliore. Un
garde-fou n'a de valeur que si son critère est celui qui compte.

## 10. Le Validated Transformation Principle est-il soutenu expérimentalement ici ?

Oui, pour l'ajustement paramétrique, avec une condition.

**Principe** : toute transformation interne susceptible de modifier un état (paramètres, structure, croyance, fragment,
primitive, stratégie, poids, code) suit `State → Transformation Proposal → Evaluation → Accept / Reject → Commit /
Rollback`. Une transformation ne peut pas détruire silencieusement un état déjà meilleur selon les critères disponibles.

**Ce qui est démontré** : appliqué à un seul étage, il restaure +47 points de récupération sur des tâches jamais vues
(IC [+38 ; +56]), sans coût de calcul supplémentaire, en ne dégradant que 4,7 % des tâches et aucune récupération.

**La condition** : le critère d'évaluation doit être aligné avec ce qui compte. Le même mécanisme avec le critère de
substitution ne restaure que 9 points. Le principe protège contre la destruction selon le critère choisi ; il ne corrige
pas un mauvais critère.

**Ce qui n'est pas démontré** : sa valeur hors de l'ajustement paramétrique, avec des atomes découverts plutôt que
fournis, ou en test confirmatoire. Il n'est implémenté qu'ici (`validated_fit`), conformément à la consigne.

### Extension possible (proposition, non réalisée)

Par ordre de proximité avec ce qui est mesuré : (1) la simplification, qui retire un atome nécessaire dans 1,6 % des
tâches et remplacerait sa règle par la même transaction ; (2) les tours de réinjection des atomes polis ; (3) la
promotion de fragments et de primitives dans les bibliothèques ; (4) les mises à jour de croyance après mesure active.
Chaque extension demanderait son propre critère observable et sa propre expérience à deux bras.

### Lien avec l'auto-évolution future (implication architecturale, pas capacité démontrée)

La même mécanique pourrait servir d'invariant à des transformations plus ambitieuses : apprentissage structurel,
consolidation, naissance et retrait de primitives, plasticité neuronale, auto-programmation, évolution architecturale.
Dans chacun de ces cas, le système modifierait son propre état et aurait besoin de ne pas perdre un état meilleur. Rien
de cela n'est démontré ici : le seul résultat est qu'un étage d'optimisation qui détruisait près de neuf bonnes solutions sur dix
cesse de le faire quand ses sorties passent par une validation. La leçon transposable est double — conserver l'état
avant de le transformer, et choisir un critère d'acceptation qui mesure réellement ce qu'on veut préserver.

## Ce que ce sous-jalon ne résout pas

- L'optimisation elle-même reste inutile sur des candidats déjà justes : le garde-fou atteint 54 à 59 %, le no-op 86 à
  89 %. La correction suivante, à tester SEULE, porterait sur l'objectif optimisé (ou sur la décision de ne pas optimiser
  un candidat issu de la synthèse), pas sur le mécanisme.
- En régime standard aux profondeurs 4–5, le gain n'est pas établi.
- Tout repose sur des atomes fournis. Avec des atomes découverts, approximatifs, l'optimisation peut être nécessaire et
  le bilan différent : non mesuré.
- 16 tâches par cellule, 64 structures par jeu.
- Rien ici n'est confirmatoire. L'hypothèse pré-enregistrée H10(b) est réfutée sur les composites ; H10(a) (vitesse)
  n'est soutenue que pour les ajustements par lots (≈ 7× par lot de 8, d'après une mesure locale ponctuelle du début du jalon ; 1,1× mesuré ici pour un ajustement isolé).

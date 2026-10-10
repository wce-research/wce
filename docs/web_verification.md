# Vérification de la refonte WCE — 3 octobre 2026

## Exécution

Stack construite et exécutée uniquement sur le Dell `<LOCAL_HOST>`, dans `~/workspace/wce`. Service unique `wce-web-1`, sain après build et après `docker compose down` puis `docker compose up -d --wait`.

Port inchangé : `127.0.0.1:3018 → web:8000`, allocation existante `wce/web`. Résolution DNS du service `web` vérifiée dans le réseau Compose. Les outils et la police source de génération des logos sont exclus de l'image ; seuls les exports et la galerie sont livrés.

## Contrôles effectués

- 161 tests ciblés `tests.test_web tests.test_wce` : **OK sur le Dell**.
- Dix pages rendues dans Chromium aux largeurs 1440, 768 et 390 px : **30 rendus sans débordement horizontal**.
- Navigation mobile, fermeture avec Échap et filtres de recherche : **OK**.
- Laboratoire → API → moteur : soulever la balle, lire la hauteur dans la même session, refuser une durée excessive : **OK**.
- Dix-sept liens/ressources internes, incluant le ZIP et les favicons : **HTTP 200**.
- Aucun échec JavaScript observé.
- 225 fichiers du kit d'identité validés : dimensions PNG/SVG, limites de poids GitHub/LinkedIn, cinq ICO multi-résolution ; archive complète créée.
- Footer commun : GitHub, X, LinkedIn et Instagram présents sur les neuf pages du site. URL officielles absentes des `.env*` audités, donc entrées « À venir » ; configuration par `WCE_*_URL` dans `.env`.

Le contrôle navigateur du site réel a utilisé des requêtes HTTP relayées par une commande SSH existante vers `127.0.0.1:3018` sur le Dell. Le serveur SSH interdit le forwarding TCP ; sa configuration n'a pas été changée. Aucun nouveau port ni stack Docker n'a été lancé sur le GEEKOM.

## Limites conservées

La suite scientifique complète n'a pas été relancée pour cette refonte visuelle. Les deux différences scientifiques entre CPU Dell et GEEKOM documentées dans le README restent ouvertes. Les tests ciblés verts ne signifient donc pas que toute la suite scientifique est verte sur le Dell.

Les comptes utilisateurs et le catalogue collaboratif ne sont pas implémentés. La structure partagée du frontend, sa frontière API et la trajectoire d'intégration sont documentées dans `hub_evolution.md`.

L'identité C2 est une direction de travail argumentée. La distinctivité et la mémorisation sont évaluées qualitativement ; elles n'ont pas fait l'objet d'un test de reconnaissance auprès d'utilisateurs ni d'une validation exhaustive des marques voisines.

## Mise à jour — 4 octobre 2026

Image reconstruite et conteneur sain sur le Dell, y compris après `docker compose down` puis `docker compose up -d --wait` ; `/healthz` répond `ok`. 163 tests ciblés HTTP/moteur passent. Le navigateur a contrôlé les 48 URL bilingues et 12 rendus représentatifs sur ordinateur/mobile : aucun débordement ni erreur JavaScript. Les 96 rendus locaux préalables couvrent également toutes les pages et les deux largeurs.

Vérifiés sur la stack réelle : survol des sous-menus, accordéons mobiles et Échap, thème nuit conservé après rechargement, changement de langue conservant la page, cookie de préférence et redirection de l’accueil, symbole WCE dans le footer et quatre marques sociales de 28 px. Le laboratoire anglais conserve la hauteur entre commandes et refuse une durée excessive.

Logs inspectés sans erreur applicative ; port limité à `127.0.0.1:3018` ; nom DNS `web` résolu dans Compose. Les URL officielles des comptes sociaux restent à renseigner. Les deux écarts scientifiques antérieurs restent ouverts.

Captures du site Docker : [navigation jour](web-nav-day.png), [page nuit](web-night.png), [navigation mobile](web-nav-mobile.png), [footer mobile](web-footer-mobile.png).

## Présentations et fidélité au backend — 4 octobre 2026

### Modifications livrées

- Cartes de mécanismes : forme arrondie, statut visible, description, chaîne conceptuelle et fichier source. Dix familles constituent l’atlas d’architecture ; six sont présentées sur l’accueil.
- Présentations des cinq rubriques et cartes vers leurs sous-pages ; cartes de poursuite dans les dossiers.
- Six nouvelles sous-pages : langage hybride, algèbre de mesure, preuves et sources, certificats génératifs, découverte active et branches. Total : 30 pages, chacune en français et anglais.
- Textes revus après lecture des interfaces et mécanismes : TransformationSpec, ModelState, TransformationPlanner, StructuralVerifier, MeasurementOperator, certificats, consolidation et continuum. Les fonctions web, les prototypes de recherche et le hub envisagé sont distingués.
- Inventaire repliable de 121 modules moteur/recherche présents sur le Dell, avec symboles supplémentaires dépliables. Il est généré par analyse statique, sans importer les modules scientifiques. `make docker-up`, `docker-build` et `docker-dev` le régénèrent avant construction.

### Vérifications

165 tests ciblés `tests.test_web tests.test_wce` passent sur le Dell : routes bilingues, liens localisés, API/session/validation, références source et actualité de l’inventaire. Les contrôles navigateur locaux couvrent 120 rendus (60 URL × ordinateur/mobile). Sur la stack réelle du Dell : 60 URL répondent, 12 rendus représentatifs sont contrôlés, les cartes ouvrent leurs dossiers, les symboles d’inventaire se déploient, et aucun débordement horizontal ni erreur JavaScript n’est observé. Les thèmes, la langue, les sous-menus, le footer et le laboratoire anglais sont également vérifiés.

Image reconstruite, conteneur sain, `/healthz` répond `ok`, logs inspectés. Le port réservé reste `127.0.0.1:3018`. Aucune stack n’a été lancée sur le GEEKOM.

### Différences entre dépôts

L’inventaire local audité sur le GEEKOM comporte 123 modules, contre 121 sur le Dell : `core/deep_structure.py` et `core/deep_world.py` sont présents uniquement sur le GEEKOM ; les symboles de `core/robust_experiments.py` diffèrent aussi. Les sources scientifiques du Dell n’ont pas été remplacées pendant ce chantier éditorial. Le site décrit son inventaire réel, généré depuis ce dépôt cible.

La suite scientifique complète n’a pas été relancée. Les deux écarts antérieurs documentés restent ouverts ; les 165 tests ciblés ne constituent pas un verdict sur l’ensemble des expériences.

### Captures du site réel

- [Architecture, vue jour](web-architecture-atlas.png)
- [Les dix cartes](web-architecture-cards.png)
- [Carte mobile en mode nuit](web-card-mobile-night.png)
- [Dossier de mécanisme sur mobile](web-mechanism-mobile.png)
- [Inventaire et symboles dépliés](web-source-map.png)

## Études interactives et vues scientifiques — 4 octobre 2026

### Livré sur le Dell

- Étude de configuration sur l’accueil : primitives, rotation et composition, repère fixe et nombre de primitives constant. Le caractère illustratif est explicite ; aucun moteur d’inférence n’est exécuté par cette interaction.
- Dix schémas de mécanismes propres aux cartes et à leurs dossiers, distincts de traces d’exécution ou de preuves expérimentales.
- Couverture prédictive/crédible par source, sur l’échelle 0–1, avec les cinq groupes de l’artefact.
- Sélecteur de neuf métriques continuum pour les neuf méthodes, moyennes, intervalles à 95 % et tableaux d’effectifs. L’ordre suit le fichier ; les coûts ne sont pas masqués par un classement général.
- Dix hypothèses de découverte active, avec quatre verdicts soutenus, trois réfutés et trois non concluants ; les champs bruts de chaque hypothèse sont dépliables.
- Provenance des vues : lien vers le JSON source, préfixe SHA-256 affiché et empreinte complète au survol. Les valeurs sont lues depuis les artefacts existants et non recopiées manuellement dans les templates.

### Vérification

168 tests HTTP/moteur passent sur le Dell. Ils comparent notamment les couvertures, moyennes, intervalles, effectifs et verdicts affichés aux JSON servis, et vérifient que le rendu suit une modification des valeurs d’un artefact.

Le script conservé dans `tests/browser/frontier.cjs` passe sur les pages réellement servies par Docker : 36 rendus français/anglais aux largeurs 1440, 768 et 390 px ; sélection des trois configurations et conservation du repère ; neuf métriques × neuf méthodes, tableaux dépliables ; champs des hypothèses ; deux thèmes et dix schémas. Aucun débordement horizontal ni erreur JavaScript observé. Le même script a également passé les contrôles sur les exports locaux avant déploiement.

Image construite uniquement sur le Dell ; conteneur `wce-web-1` sain, port inchangé `127.0.0.1:3018`. Clé SSH existante réutilisée et déjà chargée dans l’agent ; aucune modification de clé ou de configuration SSH. L’inventaire du Dell reste généré depuis ses propres modules. Les protections Docker et HTTP restent appliquées.

La suite scientifique complète n’a pas été relancée : les deux écarts antérieurs restent ouverts. Les graphiques présentent les artefacts de leurs protocoles ; ils ne démontrent pas une supériorité générale sur les LLM ou world models.

### Captures du site réel

- [Accueil jour](web-frontier-home-day.png)
- [Accueil nuit](web-frontier-home-night.png)
- [Schémas des mécanismes](web-frontier-mechanisms.png)
- [Vue continuum et intervalles](web-frontier-continuum.png)
- [Couverture par source](web-frontier-coverage.png)
- [Verdicts sur mobile](web-frontier-verdicts-mobile.png)

[Commandes de vérification navigateur](browser_checks.md).

## Révision éditoriale et frontière de publication — 4 octobre 2026

- Cartes asymétriques, atlas à grille variable, index de recherche en cartes, boutons de 44–48 px et panneaux de vision recomposés ; variantes jour/nuit et mobile conservées.
- Reproduction publique recentrée sur le moteur et les protocoles. Retrait du serveur HTTP de l’inventaire public et des références au déploiement privé dans les pages et le document scientifique.
- Dell : 171 tests ciblés verts, image reconstruite, service healthy, écoute exclusive 127.0.0.1:3018. Document privé du hub : HTTP 404.
- Navigateur : 36 rendus FR/EN desktop/tablette/mobile, schémas, données, thèmes, absence de débordement et d’erreur JavaScript.
- Export scientifique : sélection explicite sans site/hub/cloud/historique ; manifeste SHA-256 et archive déterministe. Les 155 tests du moteur passent depuis l’archive extraite séparément.
- Deux divergences scientifiques numériques préexistantes restent ouvertes ; la suite scientifique complète n’est pas déclarée verte. Licence et dépôt public officiel restent à définir. Aucun déploiement Internet ni publication GitHub effectué.

## Clarté UX et sémantique — 4 octobre 2026

172 tests ciblés verts localement et sur le Dell. Conteneur reconstruit et healthy. 54 rendus bilingues desktop/tablette/mobile validés, lexique dépliable au clavier et trois parcours de lecture vérifiés. Le débordement du fil d’Ariane mobile détecté pendant le contrôle a été corrigé ; aucun débordement ni erreur JavaScript au dernier passage. Les limites scientifiques précédemment documentées restent inchangées.

## Portée des preuves et lecture des dossiers

173 tests ciblés verts sur les deux machines ; build Docker réalisé uniquement sur le Dell, conteneur healthy. 94 rendus navigateur validés : parcours existants et dix dossiers d’architecture en FR/EN sur desktop/mobile ; trois repères de preuve par dossier, lexique au clavier, absence de débordement et d’erreur JavaScript. Aucun résultat scientifique ni seuil expérimental modifié.

## Recherche de mécanismes et liens entre dossiers

175 tests ciblés verts localement et sur le Dell. Docker reconstruit sur le Dell uniquement ; conteneur healthy, port local réservé inchangé. 94 rendus navigateur validés ; recherche par chemin de module, absence de résultat et remise à zéro contrôlées dans les deux langues. Dix dossiers, lexique au clavier, tableaux, thèmes et absence de débordement ou d’erreur JavaScript vérifiés. Aucun moteur scientifique ni résultat modifié.

## Explication des résultats et empreintes accessibles

176 tests ciblés verts localement et sur le Dell ; conteneur reconstruit et healthy. 94 rendus navigateur validés, neuf guides de métriques contrôlés à chaque sélection, empreinte complète dépliée au clavier et vérifiée sur desktop/mobile. Aucun débordement ni erreur JavaScript. Les résultats JSON n’ont pas été modifiés ; les limites numériques scientifiques préexistantes restent ouvertes.

## Découverte guidée

177 tests ciblés verts sur les deux machines. Protocole préenregistré accessible (HTTP 200), conteneur healthy sur le Dell. 94 rendus navigateur validés : titres des dix questions, filtres soutenu/réfuté/non concluant/tous avec effectifs 4/3/3/10, champs bruts conservés, contrôles au clavier, absence de débordement ou d’erreur JavaScript. Les critères et verdicts scientifiques restent inchangés.

## Inventaire recherchable et préparation des contributions

178 tests ciblés verts localement et sur le Dell ; image reconstruite, conteneur healthy sur le port local réservé. 106 rendus navigateur validés, incluant sources et contribution dans les deux langues et trois tailles d’écran. Recherche par chemin et symbole, état sans résultat, restauration des groupes à l’effacement et absence de débordement vérifiés. Aucun fichier du dépôt n’est chargé par le filtre et aucun canal de contribution opérationnel n’est annoncé.

## Laboratoire : retours d’exécution et reprise

179 tests ciblés verts sur les deux machines ; conteneur reconstruit et healthy sur le Dell. 112 rendus FR/EN desktop/tablette/mobile validés. Commandes réelles du navigateur vers le moteur : soulever à 2 m puis observer ; refus de la durée 61 s ; erreur réseau simulée, saisie conservée et reprise possible. Aucun débordement ni erreur JavaScript. Le moteur scientifique et ses protections n’ont pas été modifiés.

## Correction du contraste après inspection visuelle

Le contraste des libellés et réponses dans la console fixe sombre était incorrect en mode nuit, malgré les tests fonctionnels. Correction déployée sur le Dell après interruption du premier rebuild. Contrôle ciblé navigateur FR/EN : libellés dans les deux thèmes, couleur des réponses du moteur, commandes réelles, validation serveur, reprise après erreur réseau ; aucun débordement ni erreur JavaScript. Capture revue : docs/web-lab-mobile-verified.png. Commande de contrôle ciblé : WCE_BROWSER_SCOPE=lab avec tests/browser/frontier.cjs et les variables de connexion habituelles.

## Passe cohérente accueil / navigation / accessibilité

181 tests ciblés verts localement et sur le Dell. Contrôle des liens internes et ancres, unicité des identifiants et titre principal sur les 60 pages FR/EN. 112 rendus navigateur de régression validés : schémas, données et filtres, inventaire, guides, laboratoire réel, absence de débordement et d’erreur JavaScript. Sous-menu au clavier et au survol, fermeture Échap, menu mobile et footer à cinq rubriques contrôlés. L’accueil a également passé huit rendus dédiés FR/EN desktop/mobile jour/nuit.

Captures de revue : docs/web-home-studies-desktop.png, docs/web-footer-reviewed-mobile.png, docs/web-lab-mobile-verified.png. Conteneur healthy sur le Dell, port réservé 127.0.0.1:3018 inchangé. Aucun déploiement Internet ou dépôt GitHub public créé. Deux divergences scientifiques numériques préexistantes et la définition de la licence restent ouvertes.

La dernière capture du footer montre l’ensemble du bloc, logo et réseaux compris. Après revue, le titre français « Recherche ouverte » a été harmonisé et le lien redondant du laboratoire remplacé par ses limites. Les 13 tests de contenu passent après ces précisions ; quatre rendus ciblés FR/EN desktop/mobile du footer passent également. Image finale healthy sur le Dell.

# Vérification de la refonte WCE — 3 octobre 2026

## Exécution

Stack construite et exécutée uniquement sur le Dell `<LOCAL_HOST>`, dans `<USER_HOME>/workspace/wce`. Service unique `wce-web-1`, sain après build et après `docker compose down` puis `docker compose up -d --wait`.

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

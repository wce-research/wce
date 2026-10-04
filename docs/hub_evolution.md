# Évolution du frontend vers un hub de recherche

## Base mise en place

Le site fonctionne sans framework ni build JavaScript. Les modules ES natifs permettent d'ajouter des interfaces interactives progressivement et sont importables depuis un futur bundler. La migration vers un framework reste possible, mais elle ne constitue pas une migration automatique ni un hub déjà livré.

- `app/templates/layout.html` : layout unique, navigation, footer et chargement des assets.
- `app/templates/pages/` : contenu indépendant de chaque route. Aucun header ou footer dupliqué.
- `app/site.py` : registre de routes, rendu partagé et configuration publique validée.
- `app/static/styles.css` : tokens de couleur, typographie, grilles et comportements responsive.
- `app/static/site.js` : navigation accessible et filtres du catalogue actuel.
- `app/static/api.js` : frontière unique pour les appels JSON, chemins relatifs, cookies same-origin et gestion des erreurs.
- `app/static/app.js` : contrôleur du laboratoire ; isolé du reste du site.
- `app/web.py` : HTTP et moteur de session. Il ne contient pas de logique d'identité utilisateur.

La session du simulateur n'est **pas un compte utilisateur**. Les URL sociales sont des paramètres publics explicitement sélectionnés ; le serveur n'expose jamais l'ensemble de son environnement.

## Extension cible

| Parcours | Routes envisagées | Données et fonctionnalités nécessaires |
|---|---|---|
| Identité | `/login`, `/account`, `/u/:handle` | Authentification, profils, récupération de compte |
| Organisations | `/orgs/:slug` | Membres, rôles et permissions |
| Modèles | `/models`, `/models/:owner/:slug` | Model cards, versions, licences, artefacts |
| Architectures | `/architectures/:owner/:slug` | Source, primitives, configurations, forks |
| Datasets | `/datasets/:owner/:slug` | Dataset cards, licences, versions |
| Expériences | `/experiments/:id` | Protocoles, métriques, reproductibilité |

Ces routes sont une proposition d'évolution ; elles ne sont pas annoncées comme disponibles dans l'interface actuelle. Le registre de routes peut accueillir chaque parcours sans modifier les pages existantes. Le moteur du laboratoire conserve son API dédiée.

## Séquence d'intégration

1. Définir les ressources et contrats JSON : identifiants stables, erreurs structurées, pagination, recherche, versions et permissions. Remplacer l'index éditorial par des données persistantes lorsque le catalogue devient collaboratif.
2. Introduire une base relationnelle avec migrations, un stockage d'artefacts séparé et des tests de persistance. Réserver tout nouveau port hôte dans la gouvernance avant publication ; ces services peuvent rester internes à Compose.
3. Ajouter une authentification maintenue et éprouvée : sessions serveur durables, cookies HttpOnly, protections CSRF, contrôle des autorisations côté serveur et limitation des tentatives. Aucun jeton dans localStorage. Conserver une identité distincte de la session de simulation.
4. Ajouter les pages de comptes et catalogues dans le layout partagé. Les modules frontend utilisent `requestJSON` ; si les besoins justifient un framework, migrer les vues progressivement en conservant les URL et contrats API.
5. Ajouter les fonctions collaboratives : organisations, contributions, forks, modération et traçabilité. Les entraînements ou validations d'artefacts requièrent des workers isolés ; ne pas les exécuter dans le processus HTTP.

## Configuration du footer

Définir dans `.env` : `WCE_GITHUB_URL`, `WCE_X_URL`, `WCE_LINKEDIN_URL`, `WCE_INSTAGRAM_URL`, puis exécuter `make docker-up`. Compose injecte uniquement ces variables déclarées. Elles ne sont pas copiées dans l'image. Les URL HTTPS des domaines officiels sont acceptées ; une valeur absente ou invalide reste affichée « À venir » et ne devient pas un lien.

Aucune URL officielle n'a été trouvée dans les fichiers `.env*` audités sur le GEEKOM ou sur le Dell lors de cette refonte.

## Contenu et préférences

`app/content.py` contient les dossiers bilingues structurés ; `app/i18n.py` centralise les URL et traductions. Ces couches peuvent alimenter de futures vues sans dupliquer la navigation. Les préférences de langue et de thème restent distinctes de toute identité utilisateur. Les comptes, catalogues et entraînements communautaires restent des extensions à développer.

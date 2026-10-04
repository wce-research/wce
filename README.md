# WCE — World Configuration Engine

WCE est un moteur Python de simulation avec une interface web locale. L'application web réutilise `app.session.Session` pour les commandes, `language/` pour l'analyse et `core/` / `physics/` pour le monde et sa simulation.

## Architecture

- `app/web.py` : serveur HTTP et API `POST /api/command`, plus `GET /healthz`.
- `app/templates/` : layout partagé et 24 pages, disponibles en français et en anglais ; `app/static/` : CSS et modules ES natifs. Les appels API utilisent des chemins relatifs, sans adresse machine codée en dur.
- Site R&D multipage : accueil `/`, recherche `/research`, trois dossiers expérimentaux, architecture `/architecture`, vision `/vision`, ouverture `/open` et laboratoire `/lab`. Identité ivoire, graphite et cobalt ; assets locaux sans dépendance à un CDN. Les chiffres renvoient aux fichiers JSON et au protocole préenregistré servis sous `/research/` ; les limites sont explicites.
- [Architecture détaillée](docs/architecture_wce.md) : état opérationnel, modules exploratoires et trajectoire vers un hub ouvert.
- `app/session.py` : moteur de session déjà présent, également utilisé par le REPL.
- Un seul conteneur `web` ; pas de base de données, migration, worker, proxy ou service externe nécessaire pour ce parcours.
- Les sessions sont conservées en mémoire, isolées par cookie, pendant 24 heures au maximum et **réinitialisées après un redémarrage**. Le dépôt n'avait pas de stockage durable à migrer.
- Les modules scientifiques utilisant PyTorch sont présents dans le dépôt et couverts par des tests, mais ne sont pas chargés par le serveur web léger. Leur intégration en modèle entraînable de bout en bout reste un chantier de recherche.

## Frontend et futur hub

Voir [direction du design](docs/web_design.md) et [évolution vers un hub](docs/hub_evolution.md). Les comptes utilisateurs, catalogues collaboratifs et stockages sont des étapes futures documentées. Le site actuel n'annonce pas ces fonctions comme disponibles.

Le footer contient GitHub, X, LinkedIn et Instagram. Renseigner les URL officielles dans les variables `WCE_*_URL` de `.env` (modèle dans `.env.example`), puis `make docker-up`. Aucune URL officielle n'a été retrouvée dans les fichiers audités : les entrées restent « À venir » jusqu'à configuration.

## Démarrage sur le Dell

Prérequis : Docker Engine avec le plugin Compose. Depuis le répertoire du projet sur le Dell :

```bash
cp .env.example .env
make docker-up
make docker-ps
curl -fsS http://127.0.0.1:3018/healthz
```

Ouvrir **http://127.0.0.1:3018/** dans le navigateur du Dell. Le port hôte 3018 est réservé sous `projects.wce` dans `~/workspace/portfolio-port-governance/ports.json` (commit `7b9a6ee`) et documenté dans son tableau README. Le conteneur écoute sur son port interne 8000 ; seul 3018 est publié et uniquement sur `127.0.0.1`. L'application n'est pas exposée au LAN ou à Internet. Pour changer ce port, réserver d'abord un nouveau port dans le registre puis modifier `.env`.

**Registres vérifiés :** le registre d’autorité du GEEKOM est `<USER_HOME>/workspace/_machine_registry` (commit `140806b`). Son clone sur le Dell est `<USER_HOME>/workspace/portfolio-port-governance` (commit `7b9a6ee`). La réservation `wce/web → 3018` a été effectuée avec `./ports reserve`, validée, ajoutée aux deux tableaux README et committée dans les deux dépôts. Le clone Dell contient déjà d’autres allocations plus récentes et reste en avance sur son dépôt distant ; aucun historique non lié à WCE n’a été poussé.

```bash
make docker-down       # arrête sans supprimer d'éventuels volumes
make docker-up         # redémarre
make docker-build      # reconstruit sans cache
make docker-logs       # journaux continus
make docker-ps         # état et santé
make docker-dev        # synchronisation et redémarrage après changement de code
make test              # tests Python existants sur un environnement avec PyTorch
```

Pour exécuter la suite scientifique complète sur une installation Python indépendante :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-research.txt
make test PYTHON=.venv/bin/python
```

Le serveur web n'importe pas PyTorch et son image de production ne l'installe pas. Les tests HTTP ciblés peuvent être lancés avec `python3 -m unittest tests.test_web -v`.

État vérifié : les 3 777 tests ont été exécutés sur le Dell avec PyTorch 2.6.0+cu124 (2 échecs et 1 fichier de calibration manquant, depuis transféré). Les deux échecs ont été rejoués avec PyTorch 2.12.0+cpu dans le venv propre du projet et persistent ; ils sont sensibles à la plateforme. `tests.test_generative.SealedTests.test_hash` hache la représentation textuelle de nombres flottants, qui diffère légèrement entre CPU Dell et GEEKOM. `tests.test_language_core.SelfCorrectionTests.test_geometry_triggers_reroute` aboutit à `REJECT` sur le Dell et à `ACCEPT` sur le GEEKOM. Ces deux écarts ne sont pas contournés en modifiant les seuils ou l'empreinte scellée. Les 163 tests ciblés du moteur et du web passent sur le Dell ; la suite complète passe sur le GEEKOM avec PyTorch 2.12.0+cu130.

Le mode développement utilise `compose.dev.yaml` et Docker Compose Watch. Aucun code source n'est monté dans le conteneur de production. Le build n'intègre ni `.env`, ni clés SSH, ni résultats expérimentaux non sélectionnés (trois artefacts de recherche sont inclus explicitement). Le conteneur est exécuté par un utilisateur non root, en lecture seule, sans capacité Linux et avec une limite de taille des journaux.

## API et tests rapides

```bash
curl -fsS http://127.0.0.1:3018/healthz
curl -fsS -H 'Content-Type: application/json' \
  -d '{"command":"Où est la balle ?"}' \
  http://127.0.0.1:3018/api/command
```

Le navigateur conserve un cookie de session `HttpOnly` et `SameSite=Strict`. L'API limite la taille et la longueur des commandes, exige JSON et vérifie l'origine des requêtes navigateur. Les erreurs de commande sont renvoyées en JSON. Le service n'expose ni CORS ni authentification réseau : sa publication reste donc limitée à `localhost`.

## Étude d’identité visuelle

Ouvrir **http://127.0.0.1:3018/identity/** sur le Dell : douze concepts monochromes, cinq finalistes, évaluation qualitative, tests de tailles et déclinaisons sociales. [Dossier source](docs/identity/README.md). Le kit complet est téléchargeable depuis `/identity/wce-identity-study-01.zip`. Les propositions ne sont pas présentées comme une identité définitivement adoptée.

C2 (« Invariant décalé ») est utilisé comme identité de travail du site : symbole monochrome dans le header, favicon adaptatif clair/sombre, ICO et apple-touch-icon. Les cinq kits restent comparables dans la galerie ; ce choix de travail ne prétend pas conclure la validation de marque ni les tests de reconnaissance.

## Langues, thèmes et navigation

Les URL `/fr/` et `/en/` ouvrent les deux langues. Le sélecteur conserve la page courante ; un cookie public mémorise la langue préférée pour l’accueil. Les artefacts scientifiques sources conservent leur langue originale.

Le thème suit initialement le système. Le bouton jour/nuit conserve ensuite le choix dans `localStorage` (`wce.theme`). Aucun jeton d’authentification n’y est stocké.

Les cinq rubriques proposent des sous-pages au survol et au clavier sur ordinateur, et des boutons de déploiement sur mobile. Échap ferme le menu. Les 15 nouveaux dossiers détaillent les primitives, l’état du monde, l’apprentissage, la planification, les protocoles, les limites et la trajectoire du hub.

Le footer reprend le symbole WCE et les quatre marques sociales locales de 28 px. Renseigner les URL officielles dans les variables `WCE_*_URL` de `.env` ; sans URL, les marques restent visibles avec « À venir ».

Le laboratoire anglais reconnaît trois familles : `Lift the ball to 2 metres`, `Where is the ball?`, `Advance time by 1 second`. Les limites et validations serveur restent appliquées.

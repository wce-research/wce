# WCE Labs — identité publique et exploitation

Mise à jour du 10 octobre 2026. La direction artistique et le symbole C2 existants sont conservés.

## Audit et périmètre

Le frontend est rendu côté serveur Python (app/site.py, templates et CSS/JS), avec le laboratoire Session et l’API dans le même service web. Aucun serveur frontend supplémentaire, base de données ou worker n’est nécessaire pour cette mise à jour. Les comptes du futur hub restent une ambition et leur architecture privée n’est pas publiée.

Les mentions du projet WCE et les documents scientifiques restent WCE / World Configuration Engine. L’identité du laboratoire dans le header, footer, titres, métadonnées et nouveaux exports devient WCE Labs. Les planches historiques de recherche de logo sont conservées comme archives, pas utilisées comme identité principale.

Audit : shell partagé, routes FR/EN, liens sociaux et configuration .env.example, exports/favicon, manifeste, métadonnées et ressources de recherche. Il n’existait pas de canonical/OG/JSON-LD centralisé ; app/public_identity.py les fournit désormais. Aucun email personnel ajouté. Aucun ancien domaine introduit dans les nouveaux exports.

## Configuration

- Domaine et organisation centralisés dans app/public_identity.py : https://wcelabs.com et https://github.com/wce-research.
- PUBLIC_CONTACT_EMAIL : vide par défaut ; seule une adresse sous wcelabs.com valide est affichée. Fournir une vraie boîte avant activation.
- PUBLIC_WCE_REPO_ENABLED=false : le lien https://github.com/wce-research/wce reste masqué par défaut. Le dépôt est désormais public avec une licence Apache-2.0, selon l’état communiqué par le propriétaire ; le code scientifique local n’a pas encore été poussé. Son existence ne vaut pas publication du code. Toute publication scientifique nécessite un audit préalable.
- WCE_GITHUB_URL est conservé pour compatibilité Compose ; le frontend utilise l’organisation officielle centralisée. X, LinkedIn et Instagram gardent leur configuration existante.

## Assets et SEO

La langue canonique future du dépôt scientifique public est l’anglais : code, commentaires et docstrings, README, documentation technique, rapports, hypothèses, artefacts et métadonnées publiques. Le site reste multilingue et pourra traduire cette source canonique. Cette règle ne signifie pas que les documents locaux existants ont déjà été traduits ou publiés.

Sources : tools/build_labs_brand.py, SVG dans app/static/brand. Exports : avatars carrés PNG/SVG clair et sombre, SVG monochrome, logos horizontaux WCE Labs clair/sombre/monochrome, image sociale 1200 × 630. Favicon adaptatif SVG, ICO 16/32/48 et apple-touch-icon 180. La géométrie originale est inchangée. Régénération : python3 tools/build_labs_brand.py (Pillow et police système DejaVu Sans requis ; aucun outil ajouté à l’image de production).

Chaque page FR/EN reçoit titre WCE Labs, description prudente, canonical localisé, hreflang, OpenGraph, Twitter summary_large_image et Organization JSON-LD. Les images utilisent le domaine officiel. Le manifeste reste display=browser : aucune capacité PWA promise.

## Machines et ports

Développement : GEEKOM A6. Exécution : Dell <LOCAL_HOST>, <USER_HOME>/workspace/wce. SSH existant <OPERATOR_PROMPT> avec geekom_to_dell_ed25519 ; aucun changement de clé/configuration.

L’autorité sur le GEEKOM est ~/workspace/_machine_registry (équivalent effectif du dossier demandé). ports.json projects.wce et la table README réservent déjà web : 3018, binding 127.0.0.1. ./ports check valide le registre. Aucune nouvelle réservation nécessaire. La copie Dell ~/workspace/portfolio-port-governance est historique : ne pas remplacer l’autorité récente par cette copie.

Compose publie uniquement 127.0.0.1:3018 → web:8000 pour navigateur et API locale. Le Dell fonctionne sous Linux ; pas de couche Windows/WSL active à supposer. ss et docker compose ps ont confirmé que ce port appartient déjà à WCE, sans collision.

## Commandes sur le Dell

```bash
cd ~/workspace/wce
docker compose up -d --build --wait
docker compose ps
docker compose logs --tail=100 web
docker compose logs -f web
docker compose down
docker compose up -d --wait
```

URL locale : http://localhost:3018/fr/ ou http://localhost:3018/en/.
Contrôle : curl -fsS http://127.0.0.1:3018/healthz.
Pas d’exposition LAN ou Internet ajoutée. Les sessions du laboratoire sont temporaires ; aucun stockage persistant de comptes n’est annoncé.

## Connexion future de wcelabs.com — validation requise

Le DNS Cloudflare n’est pas modifié. Une entrée DNS ne rend pas le port privé 127.0.0.1:3018 accessible depuis Internet. Documentation : https://developers.cloudflare.com/dns/proxy-status/ et https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/.

Après autorisation explicite :

1. Auditer l’ingress actuel du Dell (reverse proxy, connecteurs existants, TLS, règles Cloudflare), choisir sa réutilisation avant toute création.
2. Valider une route HTTPS pour wcelabs.com vers WCE. Si un connecteur local existant est retenu, origine http://127.0.0.1:3018 ; si connecteur Docker, origine http://web:8000 sur un réseau partagé explicitement contrôlé. Aucun nouveau port arbitraire.
3. Contrôler Host/Origin, TLS public, politique cookies et protection de l’API interactive derrière l’ingress ; conserver les protections actuelles.
4. Valider ensemble la modification DNS/route puis l’exécuter. Ne pas créer de tunnel, redirection de box ou secret tant que cette étape n’est pas autorisée.
5. Vérifier HTTPS, FR/EN, canonical, assets OG, API et logs depuis un accès externe. Ne déclarer le domaine opérationnel qu’après ces contrôles.

Restent à fournir : adresse officielle de contact, publication du dépôt scientifique et choix/validation de l’ingress. Aucun partenariat ni crédit cloud affiché. Le frontend et l’architecture du hub ne doivent pas être inclus dans la publication du dépôt scientifique.

## Vérification de cette livraison

- Authentification GitHub SSH : compte CSAEAI ; connexion Dell confirmée. Aucun push effectué et droits d’administration de l’organisation wce-research non présumés.
- 184 tests concernés passent sur le frontend actuel (web, moteur, frontière de publication, identité). Ce contrôle n’est pas une certification de toute la recherche scientifique.
- 166 tests passent sur une copie isolée de l’arbre préparé pour le commit identité (sans les anciennes modifications non committées).
- Dell : 158 tests identité/moteur, build Docker, healthcheck, pages FR/EN, Open Research, manifeste, favicon et images HTTP 200. Arrêt/redémarrage Compose validé ; utilisateur wce ; port 3018 limité à 127.0.0.1.
- Navigateur connecté au Dell : footer FR/EN sur desktop/mobile, cinq sections, titres localisés ; aucune erreur JavaScript ni débordement. Captures de contrôle temporaires dans /tmp/wce-labs-browser sur le GEEKOM.
- Revue du diff dédié et contrôle des motifs de secrets réalisés. Aucun .env, clé privée, credential ou fichier scientifique dans le commit.
- Compose a signalé l’absence de buildx pour Bake ; le build a néanmoins réussi avec le driver Docker existant. Aucune installation système nécessaire à cette livraison.

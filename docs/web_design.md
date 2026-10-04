# WCE — direction du site de recherche

## Références consultées

- https://www.anthropic.com/research : organisation éditoriale de la recherche.
- https://openai.com/research/ : hiérarchie claire et navigation entre dossiers.
- https://amilabs.xyz/ : site AMI Labs, associé à Yann LeCun ; sobriété de la présentation scientifique.

L'identité WCE utilise un fond ivoire, du graphite et un accent cobalt. Les titres éditoriaux, règles fines, annotations monospace et diagrammes de trajectoires composent une identité dédiée au projet. Le schéma de l'accueil est une illustration conceptuelle, pas une visualisation de résultats.

## Navigation

- `/` : introduction courte et accès aux dossiers.
- `/research` : index avec filtres par domaine.
- `/research/measurement`, `/research/continual`, `/research/discovery` : dossiers et liens aux artefacts.
- `/architecture` : parcours opérationnel et modules exploratoires.
- `/vision` : les trois niveaux de maturité demandés.
- `/open` : disponibilité actuelle et étapes vers le hub ouvert.
- `/lab` : interface isolée du moteur symbolique.

Le serveur ne sert que les routes et assets explicitement autorisés. Aucun CDN, police distante ou nouveau service n'est nécessaire. Le menu mobile fonctionne au clavier et se ferme avec Échap. Les filtres exposent leur état via `aria-pressed`, le laboratoire utilise une zone de résultats annoncée aux technologies d'assistance.

Le port hôte reste `127.0.0.1:3018`, réservé pour `wce/web` dans les registres existants. La refonte n'introduit aucun port supplémentaire.

## Extension bilingue et thèmes — 4 octobre 2026

24 pages éditoriales sont accessibles sous `/fr` et `/en`. Les dossiers approfondis sont structurés dans `app/content.py` ; le catalogue `app/copy.en.json` traduit les pages historiques. Le sélecteur préserve le dossier courant. Les sources scientifiques restent inchangées.

Les tokens CSS définissent les palettes jour et nuit. `theme.js` applique le choix avant le rendu ; le système fournit la valeur initiale, puis le choix explicite persiste. Les sous-menus couvrent cinq rubriques, avec survol, clavier, Échap et accordéons mobiles.

Le symbole du footer mesure 44 px sur ordinateur ; les marques GitHub, X, LinkedIn et Instagram mesurent 28 px. Les SVG sont locaux et leurs sources/licences sont documentées dans `app/static/social/`.

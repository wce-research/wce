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

## Refonte des présentations et de la carte du backend

L’accueil présente la thèse des petites primitives et six mécanismes de recherche. L’architecture expose le parcours web puis dix cartes : composition, monde, planification, langage hybride, mesure, preuves, certificats, adaptation, découverte et branches. Des cartes dédiées rendent les sous-pages accessibles depuis chacune des cinq rubriques.

Les cartes ont une forme arrondie sobre, une hiérarchie constante (statut, mécanisme, description, chaîne conceptuelle, source), un état de focus visible et un survol discret désactivé en cas de préférence pour les mouvements réduits. Les palettes existantes ivoire/graphite/cobalt sont conservées. Les compositions géométriques sont décoratives et masquées aux lecteurs d’écran.

Trente pages sont disponibles en français et anglais. Le catalogue bilingue décrit les interfaces réellement lues dans le code, avec limites et distinctions de protocole. Le planificateur et le noyau hybride de langage sont explicitement décrits comme prototypes de recherche ; le simulateur web conserve son périmètre.

L’inventaire repliable de `/open/source` est généré à partir des fichiers présents sur la machine du dépôt. Il inclut les noms publics de classes et fonctions, sans importer les modules ni publier les environnements, identités cloud ou fichiers de résultats bruts non sélectionnés. Les symboles supplémentaires se déploient dans chaque entrée.

## Études de configuration et registres de résultats

L’illustration de l’accueil devient une étude manipulable : trois primitives, trois configurations, un repère constant et un opérateur explicite. Elle expose visuellement la thèse de composition tout en signalant son caractère purement illustratif. Aucun appel au moteur ou au réseau n’est nécessaire pour ces opérations.

Les cartes utilisent désormais dix schémas spécifiques : composition, état/action, recherche vers un objectif, vérification du langage, mesure, pondération des preuves, sondes de certificats, consolidation/retour arrière, boucle d’exploration et branches. Les dossiers distinguent ces schémas conceptuels des traces d’exécution et des résultats mesurés.

Les vues de résultats sont calculées au rendu depuis les JSON existants. La couverture est affichée par source ; les méthodes continuum se comparent sur neuf métriques, avec intervalles et tableaux ; les dix verdicts de découverte sont tous présents et dépliables. L’ordre suit les artefacts, sans classement général. Les couleurs cobalte/graphite identifient les séries, et les verdicts réfutés utilisent un accent rouge discret. Les vues sont adaptées aux deux thèmes et aux petits écrans.

## Parcours de compréhension — 4 octobre 2026

L’accueil commence par une définition de WCE et un parcours question / preuve / ambition. Les cartes présentent des questions compréhensibles avant de renvoyer vers les mécanismes et leurs sources. Le manifeste explique ce que représentent les poids et pourquoi l’architecture compte.

Chaque dossier de résultats explicite sa question, son périmètre et ce que l’expérience ne démontre pas. L’architecture propose six définitions dépliables, utilisables au clavier. Les sous-pages disposent d’un fil d’Ariane et la vision emploie des intitulés français dans la version française.

Ces composants sont rendus côté serveur, sans dépendance de recherche supplémentaire ni service du hub exposé. Les variantes anglaises sont écrites explicitement ; aucun résultat scientifique n’est traduit ou recalculé.

## Dossiers : mécanismes et portée de la preuve

Les dix dossiers d’architecture affichent trois repères : code identifié, vérification ciblée et intégration actuelle. Les cinq familles reliées aux trois artefacts publics proposent des liens vers les expériences associées, avec une mention explicite de leur portée. Les autres signalent l’absence de dossier de résultats dédié ici : aucun badge ne prétend que la seule présence de tests valide une capacité générale.

Le sommaire reçoit des cibles tactiles de 44 px et une disposition compacte sur mobile. Les exemples du laboratoire expliquent le parcours soulever / observer / avancer, en français et en anglais, sans annoncer une conversation libre. Les tests de l’interface web ne figurent plus parmi les sources scientifiques du dossier d’état du monde.

## Orientation dans l’atlas

Recherche locale des dix mécanismes par mots et chemin de module, indépendante de l’API et du moteur scientifique. La normalisation ignore casse et accents ; plusieurs mots sont combinés. Un compteur accessible et un message sans résultat accompagnent le champ. Sans JavaScript, toutes les cartes restent lisibles et aucun contrôle inactif n’apparaît.

Les liens entre dossiers sont des relations de lecture éditoriales, explicitement distinctes d’une intégration des composants. Les parcours relient transformations, monde, planification, branches, mesure, sources, certificats, apprentissage et découverte, avec accès aux expériences associées.

## Lecture des métriques et provenance

Neuf indicateurs de l’apprentissage continu disposent d’explications bilingues fondées sur `checkpoint_metrics` et les compteurs de mémoire du backend. La rétention et l’exactitude moyenne sont explicitement identifiées comme deux noms du même calcul dans ce protocole. L’oubli, le transfert arrière, l’ECE et les coûts ne sont pas confondus avec une mesure générale d’intelligence.

La couverture prédictive et la couverture crédible sont expliquées à partir du calcul d’intervalles dans `core/source_calibration.py`. Le nombre 96 est présenté comme des cas agrégés du résumé, sans l’assimiler au nombre d’exemples d’entraînement. Les types de sources ont des libellés FR/EN accessibles.

L’empreinte SHA-256 complète de chaque artefact est dépliable au clavier et sur écran tactile ; aucune information n’est disponible uniquement au survol. Les fichiers bruts et les valeurs scientifiques restent inchangés.

## Découverte : questions et verdicts

Les dix identifiants H1–H10 reçoivent des titres et questions FR/EN établis à partir du protocole préenregistré et de `core/active_benchmark.py`. Les questions exposent la comparaison étudiée ; elles ne présentent pas une hypothèse comme une capacité acquise. Pour H10, la comparaison porte sur le système complet face aux experts seuls, sans attribuer causalement le résultat à la composition seule.

Les verdicts restent lus dans `active_summary.json`, sans recalcul ni modification des seuils. Les champs exacts restent disponibles dans les panneaux. Des filtres affichent les quatre hypothèses soutenues, trois réfutées ou trois non concluantes ; le total général reste visible. Sans JavaScript, les dix hypothèses restent accessibles. Le protocole original est publié via une route de fichier explicitement autorisée, sans explorer les documents privés.

## Sources et contribution : accessibilité et statut public

L’inventaire statique se recherche par chemin de module ou symbole, avec compteur et état sans résultat. Les groupes et symboles correspondants sont ouverts pendant la recherche ; les états d’ouverture précédents sont restaurés lorsqu’elle est effacée. Le filtre ne charge aucun fichier du dépôt ni service distant. Sans JavaScript, les groupes natifs restent utilisables.

La rubrique recherche ouverte distingue ressources consultables, publication du code à finaliser et ambition de collaboration. Une préparation d’export ne signifie pas qu’un dépôt public ou une licence sont disponibles. Le canevas scientifique de contribution expose question, mécanisme, contexte, baseline, budget, critères, reproduction, artefacts et limites ; aucun formulaire d’envoi ni canal communautaire opérationnel n’est inventé. Les détails privés du site et du hub restent exclus.

## Laboratoire : exécution, erreurs et portée

Commande explicitement nommée, compteur de caractères et statut accessible. Les exemples et la saisie sont bloqués pendant l’exécution pour éviter les doubles envois ; la saisie reste disponible après un échec. Les erreurs réseau et réponses illisibles ont des messages FR/EN compréhensibles. Le délai client de 15 secondes arrête l’attente, sans prétendre annuler une action déjà exécutée : le visiteur est invité à vérifier l’état avant de renvoyer une action.

Le journal conserve au plus vingt réponses dans l’affichage, les plus récentes en premier ; cela ne réinitialise pas le monde. Toutes les réponses sont insérées comme texte. Aucune API nouvelle, aucune donnée scientifique simulée et aucune extension des capacités linguistiques ne sont introduites. Le moteur, ses validations et les limites de simulation restent inchangés.

## Accueil : privilégier les dossiers de preuve

Le teaser de six mécanismes est remplacé par trois cartes éditoriales : observation, apprentissage et découverte. Les effectifs sont lus dans les artefacts JSON, sans nombres décoratifs : cas agrégés, méthodes comparées, hypothèses soutenues/réfutées/non concluantes. Les cartes conduisent vers les protocoles et résultats ; elles n’annoncent pas un niveau d’intelligence. L’atlas des dix mécanismes conserve sa page dédiée avec un accès explicite depuis l’accueil, ce qui réduit la répétition et la longueur du parcours initial.

## Revue cohérente : orientation et accessibilité

Footer organisé en cinq rubriques et chemins courts vers preuves, mécanismes, état actuel, reproduction et laboratoire. Les cibles des thèmes et langues sont agrandies sur mobile. La navigation distingue la rubrique active de la page courante ; une rubrique parente n’est plus annoncée comme la page courante d’un sous-dossier. Les liens de cartes de recherche occupent leur largeur sur mobile.

Contrôle systématique des trente pages dans les deux langues : liens internes, ancres, unicité des identifiants et titre principal. La revue des textes des templates ne trouve pas de texte français sans traduction anglaise dans les pages servies. Les changements du site restent séparés des sources scientifiques et de l’architecture privée du hub.

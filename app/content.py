"""Bilingual research dossiers. Structured content can later come from a hub API."""
from html import escape

# title, introduction, maturity, sections, source entry points, related routes
DOSSIERS = {}

def dossier(route, fr, en, sources=(), related=()):
    DOSSIERS[route] = {'fr':fr, 'en':en, 'sources':sources, 'related':related}

dossier('/research/methodology',
 ('Méthode & protocoles','Relier chaque affirmation à une expérience et à un critère annoncé.','Méthode',[
 ('Avant de mesurer','Fixer la question, les baselines, les graines et les critères de succès avant exécution. Le préenregistrement des certificats génératifs distingue huit hypothèses et leurs seuils : ce protocole ne doit pas être réécrit après lecture des résultats.'),
 ('Comparer à budget visible','Mémoire, nombre de paramètres, nombre de requêtes et coût d’entraînement sont des axes distincts. Une exactitude plus élevée ne suffit pas à conclure si les budgets diffèrent ; les fichiers continuum exposent plusieurs de ces dimensions.'),
 ('Publier l’unité statistique','Le protocole des certificats utilise la graine comme unité statistique, avec au moins 16 graines et des intervalles à 95 %. Ne pas confondre ces graines avec les 96 observations du résumé de couverture prédictive.'),
 ('Conserver les verdicts','Les hypothèses soutenues, réfutées et non concluantes appartiennent au même dossier. Un verdict est attaché à son protocole, pas à une supériorité générale de WCE.')]),
 ('Methods & protocols','Connect each claim to an experiment and a declared criterion.','Method',[
 ('Before measurement','Fix the question, baselines, seeds and success criteria before execution. The generative certificates preregistration distinguishes eight hypotheses and their thresholds; the protocol must not be rewritten after inspecting results.'),
 ('Compare with visible budgets','Memory, parameter count, query count and training cost are separate axes. Higher accuracy is insufficient when budgets differ; continuum artifacts expose several of these dimensions.'),
 ('Publish the statistical unit','The certificates protocol uses seeds as the statistical unit, with at least 16 seeds and 95% intervals. These seeds are not the 96 observations in the predictive coverage summary.'),
 ('Preserve the verdicts','Supported, refuted and inconclusive hypotheses belong in the same record. A verdict applies to its protocol, not to general superiority of WCE.')]),
 ('docs/hypotheses_generative_certificates.md','scripts/continuum_seed.py','scripts/continuum_aggregate.py'),('/research/continual','/research/limitations'))

dossier('/research/limitations',
 ('Limites & résultats négatifs','Ce que les artefacts actuels permettent de dire — et ce qu’ils ne permettent pas encore d’établir.','État actuel',[
 ('La portée des résultats','Les mesures sont spécifiques aux tâches et aux distributions testées. Ni la couverture prédictive ni la rétention sur un flux expérimental ne constituent une mesure d’intelligence générale.'),
 ('Les hypothèses réfutées','Dans active_summary.json, H2, H3 et H8 sont réfutées ; H1, H4 et H7 sont non concluantes. Ces identifiants appartiennent au protocole de cet artefact, indépendamment des autres documents d’hypothèses.'),
 ('La sensibilité numérique','Deux tests scientifiques produisent encore des résultats différents entre le CPU du Dell et le GEEKOM : une empreinte fondée sur des flottants et une décision de reroutage. Ces écarts ne sont pas masqués en changeant les seuils ou le hash attendu.'),
 ('La comparaison encore absente','Les artefacts exposés ne démontrent aucune supériorité générale sur les LLM ou les world models. Une comparaison nécessitera tâches communes, budgets transparents, baselines et protocole reproductible.')]),
 ('Limitations & negative results','What current artifacts can support, and what they cannot yet establish.','Current state',[
 ('Scope of the results','Measurements belong to the tested tasks and distributions. Predictive coverage and retention on an experimental stream are not measurements of general intelligence.'),
 ('Refuted hypotheses','In active_summary.json, H2, H3 and H8 are refuted; H1, H4 and H7 are inconclusive. Their identifiers belong to that artifact’s protocol, independently of other hypothesis documents.'),
 ('Numerical sensitivity','Two scientific tests still differ between the Dell CPU and GEEKOM: a floating-point-based fingerprint and a rerouting decision. These differences are not concealed by changing thresholds or the expected hash.'),
 ('The missing comparison','Published artifacts do not establish general superiority over LLMs or world models. Such a comparison needs common tasks, visible budgets, baselines and a reproducible protocol.')]),
 ('results/active_summary.json','results/continuum_summary.json','tests/test_generative.py','tests/test_language_core.py'),('/research/discovery','/research/methodology'))

dossier('/architecture/primitives',
 ('Primitives, transformations, invariants','Quelques briques explicites plutôt qu’un symbole qui prétend représenter toute l’intelligence.','Prototype & exploration',[
 ('Représenter','Le parcours opérationnel représente une balle, un support et leurs propriétés. Les entités, relations et contraintes vivent dans core/world.py ; elles sont inspectables dans le code.'),
 ('Transformer','Une commande devient une action appliquée au monde. Le moteur d’exécution et la simulation sont distincts : une transformation immédiate n’est pas le même mécanisme que la progression d’un système physique.'),
 ('Contraindre','Le prototype impose notamment une hauteur non négative. Les modules invariants et invariant_algebra explorent plus largement la représentation d’invariants ; leur présence n’implique pas une intégration complète au laboratoire web.'),
 ('Composer','La thèse WCE est qu’un petit ensemble de primitives et de transformations peut engendrer des structures complexes. C’est un programme de recherche : sa portée doit être mesurée par des tâches et des expériences définies.')]),
 ('Primitives, transformations, invariants','A small set of explicit building blocks, rather than a symbol claiming to explain all intelligence.','Prototype & exploration',[
 ('Represent','The operational pathway represents a ball, a support and their properties. Entities, relations and constraints live in core/world.py and can be inspected in the source.'),
 ('Transform','A command becomes an action applied to the world. Execution and simulation are separate: an immediate transformation is not the same mechanism as the progression of a physical system.'),
 ('Constrain','The prototype enforces a non-negative height, among other constraints. The invariants and invariant_algebra modules explore broader invariant representations; their presence does not imply full integration into the web laboratory.'),
 ('Compose','The WCE thesis is that a small set of primitives and transformations can generate complex structures. This is a research programme whose scope must be measured through defined tasks and experiments.')]),
 ('core/world.py','core/executor.py','core/invariants.py','core/invariant_algebra.py'),('/architecture/world-state','/architecture/learning'))

dossier('/architecture/world-state',
 ('État du monde & simulation','Un état explicite, des actions et une évolution observable.','Opérationnel',[
 ('Le monde initial','Chaque session construit une balle et un sol, avec position, vitesse, masse et relation de support. Aucun monde partagé entre visiteurs n’est annoncé : les sessions du laboratoire sont isolées par cookie.'),
 ('Action et observation','Soulever la balle modifie sa hauteur ; une question lit l’état observé. L’interface utilise le même moteur que le REPL. La réponse est produite par des mécanismes symboliques, sans appel à un LLM externe.'),
 ('Progression du temps','Le moteur physique applique ses lois lors d’une commande temporelle. La durée maximale d’une requête web est de 60 secondes pour limiter le travail effectué dans le processus HTTP.'),
 ('Cycle de vie','L’état reste en mémoire, avec expiration des sessions et capacité limitée. Un redémarrage réinitialise les mondes. Un futur hub devra séparer comptes durables, artefacts stockés et simulations temporaires.')]),
 ('World state & simulation','An explicit state, actions and observable evolution.','Operational',[
 ('The initial world','Each session constructs a ball and a ground support, with position, velocity, mass and a support relation. No shared visitor world is claimed: laboratory sessions are isolated by cookie.'),
 ('Action and observation','Lifting the ball changes its height; a question reads the observed state. The interface uses the same engine as the REPL. Answers are generated symbolically, without calling an external LLM.'),
 ('Time progression','The physics engine applies its laws when a temporal command is executed. Each web request is limited to 60 seconds of simulation to bound work in the HTTP process.'),
 ('Lifecycle','State is held in memory, with session expiry and a capacity limit. Restarting resets worlds. A future hub must separate durable accounts, stored artifacts and temporary simulations.')]),
 ('app/session.py','core/world.py','physics/engine.py','language/answer.py'),('/lab','/lab/limits'))

dossier('/architecture/learning',
 ('Apprentissage & plasticité','Adapter les structures, retenir les compétences et rendre leurs coûts mesurables.','Exploration',[
 ('Trois questions distinctes','Acquérir une nouvelle compétence, préserver une ancienne compétence et limiter le coût mémoire ne sont pas une seule métrique. Les expériences continuum mesurent plusieurs dimensions pour éviter de réduire l’apprentissage à une moyenne finale.'),
 ('Les mécanismes explorés','Plasticité, croissance structurelle, replay et certificats génératifs sont étudiés dans des modules et expériences spécifiques. Ils ne constituent pas encore un parcours d’entraînement commun intégré au site.'),
 ('L’articulation à construire','Des interfaces stables doivent préciser états, objectifs, mises à jour, contraintes et observables. Cette intégration est nécessaire avant d’annoncer qu’un utilisateur peut entraîner un modèle WCE frontier complet depuis le navigateur.')]),
 ('Learning & plasticity','Adapt structures, retain skills and make their costs measurable.','Exploration',[
 ('Three separate questions','Acquiring a new skill, preserving an old one and limiting memory costs are not one metric. Continuum experiments measure several dimensions instead of reducing learning to a final mean.'),
 ('Mechanisms being explored','Plasticity, structural growth, replay and generative certificates are studied in specific modules and experiments. They do not yet form a shared training pathway integrated into this website.'),
 ('Integration still required','Stable interfaces must define states, objectives, updates, constraints and observables. This integration is required before claiming users can train a complete frontier WCE model from a browser.')]),
 ('core/plasticity.py','core/safe_plasticity.py','core/continuum.py','results/continuum_summary.json'),('/research/continual','/open/reproduce'))

dossier('/architecture/planning',
 ('Raisonnement & planification','Relier observations, état du monde et conséquences d’actions.','Exploration',[
 ('Langage ancré','Le parcours web interprète une grammaire limitée puis agit sur un monde explicite. Les modules grounded, discourse et language_bridge explorent d’autres liens entre langage et représentation.'),
 ('Prévoir et choisir','La planification demande d’évaluer des actions possibles à partir d’un état et d’un objectif. Les modules unified_agent et active_world font partie de l’exploration ; le laboratoire web ne prétend pas offrir un planificateur général intégré.'),
 ('Conditions d’évaluation','Un protocole devra exposer objectifs, horizon, transitions, contraintes et budget de recherche. Décrire un mécanisme de planification n’établit pas son efficacité sur des tâches inconnues.')]),
 ('Reasoning & planning','Connect observations, world state and the consequences of actions.','Exploration',[
 ('Grounded language','The web pathway interprets a limited grammar and acts on an explicit world. The grounded, discourse and language_bridge modules explore other links between language and representation.'),
 ('Predict and choose','Planning requires evaluating possible actions from a state and objective. The unified_agent and active_world modules are exploratory; the web laboratory does not claim to offer an integrated general planner.'),
 ('Evaluation conditions','A protocol must expose objectives, horizon, transitions, constraints and search budget. Describing a planning mechanism does not establish its effectiveness on unfamiliar tasks.')]),
 ('core/grounded.py','core/discourse.py','core/language_bridge.py','core/unified_agent.py','core/active_world.py'),('/research/discovery','/vision/exploration'))

dossier('/vision/now',
 ('Ce qui existe aujourd’hui','Le socle vérifiable de WCE.','État actuel',[
 ('Un moteur opérationnel','Le dépôt contient un moteur symbolique Python, une simulation physique simple, un REPL et le laboratoire web. Les commandes utilisent le même moteur de session.'),
 ('Des expériences consultables','Trois artefacts JSON sont exposés : couverture prédictive, apprentissage continu et découverte active. Les résultats restent spécifiques à leurs protocoles et incluent des limites et des verdicts négatifs.'),
 ('Un environnement local','Le site est construit et exécuté par Docker Compose sur le Dell, sur un unique port réservé. Les composants de recherche existent dans le dépôt ; ils ne sont pas tous chargés par le serveur web.')]),
 ('What exists now','The verifiable foundation of WCE.','Current state',[
 ('An operational engine','The repository contains a Python symbolic engine, a simple physics simulation, a REPL and the web laboratory. Commands use the same session engine.'),
 ('Inspectable experiments','Three JSON artifacts are exposed: predictive coverage, continual learning and active discovery. Results remain specific to their protocols and include limitations and negative verdicts.'),
 ('A local environment','Docker Compose builds and runs the site on the Dell using one reserved port. Research components exist in the repository; not all are loaded by the web server.')]),
 ('app/session.py','app/repl.py','compose.yaml'),('/architecture','/research'))

dossier('/vision/exploration',
 ('Ce qui est exploré','Définir des questions avant d’annoncer des capacités.','Exploration',[
 ('Architecture et primitives','Quelles primitives, transformations et invariants permettent une composition utile ? Leur simplicité formelle doit être confrontée à la richesse des tâches et au coût de l’exécution.'),
 ('Apprentissage et observation','Comment une structure évolue-t-elle, que retient-elle et comment observer ses changements ? Plasticité, apprentissage structurel et mesure constituent des axes complémentaires.'),
 ('État, raisonnement, actions','Comment ancrer le langage, représenter un état du monde et choisir une séquence d’actions ? Les modules sont exploratoires et leurs interfaces communes restent à consolider.')]),
 ('What is being explored','Define questions before claiming capabilities.','Exploration',[
 ('Architecture and primitives','Which primitives, transformations and invariants support useful composition? Their formal simplicity must be tested against task complexity and execution cost.'),
 ('Learning and observation','How does a structure evolve, what does it retain and how can its changes be observed? Plasticity, structural learning and measurement are complementary research directions.'),
 ('State, reasoning, actions','How can language be grounded, world state represented and action sequences selected? Modules are exploratory and their common interfaces still need consolidation.')]),
 ('core/active_structure.py','core/measurement.py','core/unified_agent.py'),('/architecture/primitives','/architecture/planning'))

dossier('/vision/frontier-hub',
 ('Un futur hub d’architectures','Publier les mécanismes, les expériences et les moyens de les prolonger.','Ambition',[
 ('Les ressources du futur hub','Architectures, modèles, datasets et expériences devront disposer de fiches, versions, licences et artefacts. Ces catalogues ne sont pas encore des services collaboratifs disponibles sur ce site.'),
 ('Les comptes et les organisations','Profils, organisations, rôles et permissions nécessitent une authentification maintenue, des sessions durables et des contrôles côté serveur. Le cookie du simulateur n’est pas un compte utilisateur.'),
 ('L’entraînement reproductible','L’objectif est de permettre à chacun d’améliorer, de forker et d’entraîner ses propres modèles. Il faudra des contrats d’entraînement, des données versionnées, des workers isolés et des comparaisons reproductibles.'),
 ('Le principe éditorial','Ouvrir l’architecture implique de rendre inspectables les mécanismes autant que les poids. L’ambition de plateforme ne transforme pas les résultats actuels en preuve de supériorité.')]),
 ('A future architecture hub','Publish mechanisms, experiments and the means to extend them.','Ambition',[
 ('Future hub resources','Architectures, models, datasets and experiments will need cards, versions, licences and artifacts. These catalogues are not yet collaborative services available on this site.'),
 ('Accounts and organisations','Profiles, organisations, roles and permissions require maintained authentication, durable sessions and server-side checks. The simulator cookie is not a user account.'),
 ('Reproducible training','The goal is to enable anyone to improve, fork and train their own models. It requires training contracts, versioned data, isolated workers and reproducible comparisons.'),
 ('The editorial principle','Opening architecture means making mechanisms as inspectable as weights. Platform ambition does not turn current results into evidence of superiority.')]),
 (),('/open/contribute','/open/reproduce'))

dossier('/open/reproduce',
 ('Reproduire & exécuter','Un parcours web reproductible ; des expériences scientifiques à exécuter dans leur propre environnement.','Disponible localement',[
 ('Démarrer le site','Sur le Dell, depuis le dépôt WCE : make docker-up, puis make docker-ps. L’interface est accessible sur http://127.0.0.1:3018/. Le port est réservé dans la gouvernance ; aucun autre service n’est publié.'),
 ('Reconstruire et observer','make docker-build reconstruit les images ; make docker-logs suit les journaux ; make docker-down arrête la stack. Les sessions de simulation sont réinitialisées après redémarrage, car elles ne sont pas persistantes.'),
 ('Distinguer les environnements','L’image du site ne charge pas PyTorch. Les expériences scientifiques requièrent l’environnement de recherche du dépôt, ses dépendances et ses fixtures. Faire tourner le site n’équivaut pas à reproduire un benchmark.'),
 ('Consigner le contexte','Noter versions, plateforme, graines, budget, script et artefact de sortie. Les deux divergences numériques connues entre le Dell et le GEEKOM sont conservées dans la documentation.')]),
 ('Reproduce & run','A reproducible web pathway; scientific experiments run in their own research environment.','Available locally',[
 ('Start the site','On the Dell, from the WCE repository: make docker-up, then make docker-ps. Open http://127.0.0.1:3018/. This port is reserved in the governance registry; no other service is published.'),
 ('Rebuild and observe','make docker-build rebuilds images; make docker-logs follows logs; make docker-down stops the stack. Simulation sessions reset on restart because they are not persistent.'),
 ('Separate the environments','The website image does not load PyTorch. Scientific experiments require the repository’s research environment, dependencies and fixtures. Running the site does not reproduce a benchmark.'),
 ('Record context','Record versions, platform, seeds, budget, script and output artifact. The two known numerical differences between the Dell and GEEKOM remain documented.')]),
 ('Makefile','Dockerfile','compose.yaml','requirements-research.txt'),('/research/methodology','/lab/limits'))

dossier('/open/source',
 ('Code source & artefacts','Relier le mécanisme, le protocole et le résultat.','Disponible localement',[
 ('Lire les mécanismes','Le code du moteur est dans app, core, language et physics. Les points d’entrée de chaque dossier identifient les modules pertinents, plutôt que d’annoncer un nouveau code source parallèle.'),
 ('Lire les résultats','Les fichiers JSON sont servis directement sous /research/. Le document d’architecture et le protocole des certificats sont accessibles comme sources en français. La traduction de l’interface ne modifie pas les artefacts scientifiques.'),
 ('Préparer la publication','L’URL publique officielle du dépôt et la licence restent à formaliser. Le lien GitHub du footer est activé uniquement lorsqu’une URL officielle est configurée ; aucune adresse de projet n’est inventée.')]),
 ('Source code & artifacts','Connect the mechanism, protocol and result.','Available locally',[
 ('Read the mechanisms','The engine source lives in app, core, language and physics. Entry points in each note identify relevant modules instead of claiming a parallel implementation.'),
 ('Read the results','JSON files are served directly under /research/. The architecture document and certificates protocol are available as French source documents. Translating the interface does not modify scientific artifacts.'),
 ('Prepare the public release','The official public repository URL and licence still need formalisation. The footer’s GitHub link is activated only when an official URL is configured; no project address is invented.')]),
 ('docs/architecture_wce.md','results/predictive_coverage.json','results/continuum_summary.json','results/active_summary.json'),('/research','/open/contribute'))

dossier('/open/contribute',
 ('Contribuer & forker','Construire les conditions d’une collaboration scientifique ouverte.','Trajectoire',[
 ('Une contribution vérifiable','Décrire la question, le mécanisme modifié, la baseline, les coûts et le protocole. Joindre code, tests et artefacts ; une nouvelle capacité doit pouvoir être inspectée et reproduite.'),
 ('Conserver l’historique expérimental','Un fork doit préciser versions, dépendances, données et écarts de comportement. Les résultats négatifs comptent ; ne pas réécrire les critères après avoir observé les mesures.'),
 ('Ce qui reste à formaliser','Licence, dépôt public, guide de contribution, revue des changements et règles de publication sont à établir. Cette page ne prétend pas fournir aujourd’hui un compte, un système de pull requests ou un service d’upload.')]),
 ('Contribute & fork','Build the conditions for open scientific collaboration.','Roadmap',[
 ('An inspectable contribution','Describe the question, changed mechanism, baseline, costs and protocol. Include code, tests and artifacts; a new capability must be inspectable and reproducible.'),
 ('Preserve experimental history','A fork should specify versions, dependencies, data and behavioural differences. Negative results matter; do not rewrite criteria after inspecting measurements.'),
 ('Still to formalise','Licensing, a public repository, contribution guidance, change review and publication rules remain to be established. This page does not claim to provide accounts, pull requests or an upload service today.')]),
 (),('/research/methodology','/vision/frontier-hub'))

dossier('/lab/guide',
 ('Guide des commandes','Une grammaire explicite et volontairement limitée.','Opérationnel',[
 ('Soulever','Français : « Soulève la balle à 2 mètres ». Anglais : « Lift the ball to 2 metres ». La commande modifie la hauteur de la balle dans votre session ; la réponse de succès est OK.'),
 ('Observer','Français : « Où est la balle ? ». Anglais : « Where is the ball? ». La réponse lit l’état actuel ; une session neuve commence avec une balle sur le sol.'),
 ('Faire progresser','Français : « Avance le temps de 1 seconde ». Anglais : « Advance time by 1 second ». La durée doit être positive ou nulle et ne peut pas dépasser 60 secondes par requête.'),
 ('Rester dans le périmètre','L’anglais du laboratoire est un adaptateur pour ces trois familles de commandes. Il ne constitue pas un nouveau modèle de langage ni une grammaire anglaise générale.')]),
 ('Command guide','An explicit and deliberately limited grammar.','Operational',[
 ('Lift','French: “Soulève la balle à 2 mètres”. English: “Lift the ball to 2 metres”. The command changes the ball’s height in your session; successful execution returns OK.'),
 ('Observe','French: “Où est la balle ?”. English: “Where is the ball?”. The answer reads the current state; a new session begins with the ball on the ground.'),
 ('Advance','French: “Avance le temps de 1 seconde”. English: “Advance time by 1 second”. Duration must be non-negative and cannot exceed 60 seconds per request.'),
 ('Stay within scope','English support in the laboratory is an adapter for these three command families. It is not a new language model or a general English grammar.')]),
 ('language/command.py','app/localization.py'),('/lab','/lab/protocol'))

dossier('/lab/protocol',
 ('Protocole interactif','Une expérience minimale pour vérifier les transitions et l’isolation des sessions.','Opérationnel',[
 ('Étape 01 — état initial','Ouvrir le laboratoire dans une nouvelle session et demander la position de la balle. Elle doit se trouver sur le sol avant toute action.'),
 ('Étape 02 — transformation','Soulever la balle à 2 mètres, puis lire à nouveau sa position. Le résultat attendu est une hauteur de 2 mètres ; les deux commandes utilisent le même cookie de session.'),
 ('Étape 03 — simulation','Faire progresser le temps puis observer l’état. Le changement provient des lois du moteur physique. Cette manipulation montre un mécanisme explicite, pas une découverte autonome de physique.'),
 ('Étape 04 — isolation','Une autre session de navigateur doit retrouver un monde initial indépendant. Un redémarrage du service réinitialise toutes les sessions ; ce comportement est documenté et testé.')]),
 ('Interactive protocol','A minimal experiment to inspect transitions and session isolation.','Operational',[
 ('Step 01 — initial state','Open the laboratory in a new session and ask for the ball’s position. It should be on the ground before any action.'),
 ('Step 02 — transformation','Lift the ball to 2 metres, then ask for its position again. Expected result: a height of 2 metres; both commands use the same session cookie.'),
 ('Step 03 — simulation','Advance time and observe the state. The change comes from the physics engine’s laws. This demonstrates an explicit mechanism, not autonomous discovery of physics.'),
 ('Step 04 — isolation','A separate browser session should have an independent initial world. Restarting the service resets all sessions; this behaviour is documented and tested.')]),
 ('tests/test_web.py','tests/test_wce.py','app/session.py'),('/lab','/lab/limits'))

dossier('/lab/limits',
 ('Limites du laboratoire','Comprendre le périmètre avant d’interpréter une démonstration.','État actuel',[
 ('Un monde réduit','Le laboratoire manipule une balle et un support. Il ne représente pas un environnement réel complet ni un système entraînable de bout en bout.'),
 ('Une interface bornée','Les requêtes sont limitées à 4 096 octets, les commandes à 500 caractères et la simulation à 60 secondes par requête. Les entrées sont validées côté serveur et les origines navigateur sont contrôlées.'),
 ('Pas de compte utilisateur','Le cookie isole une session de simulation en mémoire. Il n’est pas une identité, un profil ou un espace personnel durable. Le futur hub devra fournir ces services séparément.'),
 ('Une exposition locale','La stack est accessible sur localhost du Dell. Aucun tunnel public ou port de base de données n’est publié. Cette configuration ne constitue pas un déploiement Internet multi-utilisateur.')]),
 ('Laboratory limitations','Understand scope before interpreting a demonstration.','Current state',[
 ('A reduced world','The laboratory manipulates a ball and a support. It does not represent a complete real-world environment or an end-to-end trainable system.'),
 ('A bounded interface','Requests are limited to 4,096 bytes, commands to 500 characters and simulation to 60 seconds per request. Inputs are validated on the server and browser origins are checked.'),
 ('Not a user account','The cookie isolates an in-memory simulation session. It is not an identity, profile or durable workspace. A future hub must provide these services separately.'),
 ('Local exposure','The stack is accessible on the Dell’s localhost. No public tunnel or database port is published. This is not an Internet deployment for multiple users.')]),
 ('app/web.py','compose.yaml'),('/lab/guide','/vision/frontier-hub'))


def render_dossier(route, language):
    from app.i18n import localized_path
    record=DOSSIERS[route]; title,intro,status,sections=record[language]
    category='/' + route.split('/')[1]
    labels={'fr':('Sommaire','Points d’entrée dans le dépôt','Pour poursuivre','Retour à la rubrique'), 'en':('On this page','Repository entry points','Continue exploring','Back to the section')}[language]
    toc=''.join(f'<a href="#section-{i}">{escape(name)}</a>' for i,(name,_) in enumerate(sections,1))
    article=''.join(f'<section id="section-{i}"><span class="eyebrow">0{i}</span><h2>{escape(name)}</h2><p>{escape(text)}</p></section>' for i,(name,text) in enumerate(sections,1))
    sources=''.join(f'<li><code>{escape(source)}</code></li>' for source in record['sources'])
    resources=f'<section class="dossier-sources"><h2>{labels[1]}</h2><ul>{sources}</ul></section>' if sources else ''
    from app.site import page_title
    related=''.join(f'<a href="{localized_path(url,language)}">{escape(page_title(url,language))} ↗</a>' for url in record['related'])
    return f'<section class="page-head"><p class="eyebrow">WCE / {escape(status)}</p><h1>{escape(title)}</h1><p class="lead">{escape(intro)}</p><a class="text-link" href="{localized_path(category,language)}">← {labels[3]}</a></section><div class="dossier-layout"><aside class="dossier-toc"><p class="eyebrow">{labels[0]}</p>{toc}</aside><article class="dossier-body">{article}{resources}<section class="dossier-related"><h2>{labels[2]}</h2>{related}</section></article></div>'

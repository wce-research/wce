"""Audited, bilingual map of research modules; never imports scientific runtimes."""
from html import escape

FAMILIES = [
    dict(key='composition', route='/architecture/primitives', status='research', sources=('core/transformations.py','core/invariants.py','core/transform_graph.py'), tests=('tests/test_transformations.py',),
         fr=('Composer des transformations','Une opération déclare ce qu’elle change, ce qu’elle préserve et comment elle se compose.','Configuration · opérateur · invariant',[
             ('Une opération inspectable','TransformationSpec expose domaine, paramètres, coût, préconditions et postconditions. Les transformations géométriques, composites et macros fournissent des objets manipulables dans le code.'),
             ('La composition comme objet','CompositeTransformation et MacroTransformation organisent des séquences réutilisables. TransformGraph enregistre les transformations ; ce graphe rend une trajectoire inspectable, sans constituer une explication exhaustive d’un modèle.'),
             ('Des invariants explicites','InvariantSet décrit les propriétés déclarées préservées. Une déclaration doit être confrontée à l’exécution et aux tests : elle ne constitue pas une garantie universelle.')]),
         en=('Compose transformations','An operation declares what it changes, preserves and how it composes.','Configuration · operator · invariant',[
             ('An inspectable operation','TransformationSpec exposes domain, parameters, cost, preconditions and postconditions. Geometric, composite and macro transformations are objects that can be manipulated in code.'),
             ('Composition as an object','CompositeTransformation and MacroTransformation organise reusable sequences. TransformGraph records transformations, making a trajectory inspectable without providing an exhaustive model explanation.'),
             ('Explicit invariants','InvariantSet describes properties declared to be preserved. Declarations must be checked against execution and tests; they are not universal guarantees.')])),
    dict(key='world', route='/architecture/world-state', status='web', sources=('app/session.py','core/world.py','physics/engine.py'), tests=('tests/test_wce.py',),
         fr=('Faire évoluer un monde','Entités, propriétés, relations et contraintes forment l’état que les commandes transforment.','État · action · simulation',[
             ('Le monde du laboratoire','Session construit une balle et un sol. Position, vitesse, masse et relation de support sont explicites. Chaque navigateur possède une session distincte, temporaire et conservée en mémoire.'),
             ('Deux mécanismes distincts','Soulever la balle applique une action immédiate. Avancer le temps appelle la simulation, avec un pas de 0,1 seconde, une loi de gravité et une interaction de collision avec le sol.'),
             ('Ce que la démonstration établit','Une commande peut modifier un état puis une question peut le lire. Le parcours web n’importe pas les modules scientifiques PyTorch et ne découvre pas les lois physiques utilisées.')]),
         en=('Evolve a world','Entities, properties, relations and constraints form the state changed by commands.','State · action · simulation',[
             ('The laboratory world','Session constructs a ball and a ground support. Position, velocity, mass and support relations are explicit. Each browser has a separate, temporary in-memory session.'),
             ('Two distinct mechanisms','Lifting the ball applies an immediate action. Advancing time calls simulation with a 0.1-second step, a gravity law and a ground collision interaction.'),
             ('What the demo establishes','A command can change state and a question can read it. The web pathway does not import PyTorch research modules or discover the physical laws it uses.')])),
    dict(key='planning', route='/architecture/planning', status='research', sources=('core/reasoning.py','core/planner.py'), tests=('tests/test_reasoning_core.py','tests/test_planner.py'),
         fr=('Raisonner et planifier','Des faits, des conditions et des objectifs guident une recherche bornée de compositions.','Faits · objectifs · recherche',[
             ('Raisonner sous conditions','FactBase, RuleEngine et ReasoningContext distinguent faits, règles, hypothèses et préconditions inconnues. Le code expose les contradictions et les étapes d’inférence.'),
             ('Chercher un chemin','TransformationPlanner cherche des séquences de transformations vers un Goal. Il propose recherche en largeur, en profondeur et à coût uniforme ; StateSignature identifie les états rencontrés.'),
             ('Rendre le budget visible','SearchMetrics et SearchResult rendent compte de la recherche. Les stratégies et politiques d’hypothèses sont paramétrables ; leur existence ne démontre pas une capacité générale de planification.'),
             ('Intégration actuelle','Ce planificateur existe dans le dépôt et ses tests. L’API du laboratoire exécute des commandes directes : elle ne fournit pas de parcours de planification générale.')]),
         en=('Reason and plan','Facts, conditions and goals guide a bounded search over compositions.','Facts · goals · search',[
             ('Conditional reasoning','FactBase, RuleEngine and ReasoningContext distinguish facts, rules, assumptions and unknown preconditions. Contradictions and inference steps are explicit.'),
             ('Search for a path','TransformationPlanner searches transformation sequences towards a Goal. It provides breadth-first, depth-first and uniform-cost search; StateSignature identifies encountered states.'),
             ('Expose the budget','SearchMetrics and SearchResult describe the search. Strategies and assumption policies are configurable; their existence does not establish general planning ability.'),
             ('Current integration','The planner exists in the repository and its tests. The laboratory API executes direct commands, without providing a general planning workflow.')])),
    dict(key='language', route='/architecture/language', status='research', sources=('core/language_core.py','core/language_bridge.py','language/command.py'), tests=('tests/test_language_core.py',),
         fr=('Ancrer le langage dans l’action','Un prototype hybride relie candidats neuronaux, analyse structurée et vérification avant décision.','Texte · structure · décision',[
             ('Un prototype hybride existant','Compact WCE Language Model utilise un petit cœur neuronal au niveau caractère, une analyse de clauses, des entités et des relations. Il peut construire des objectifs et programmes de transformations.'),
             ('Vérifier avant de décider','StructuralVerifier et AntiHallucinationPolicy alimentent des décisions ACCEPT, VERIFY, REROUTE, ABSTAIN, REJECT et ASK. Ces noms désignent une politique expérimentale, pas une garantie d’absence d’hallucination.'),
             ('Le périmètre d’apprentissage','Le corpus est artificiel et contrôlé. Le prototype ne couvre pas le français réel dans son ensemble et les résultats ne démontrent pas une alternative générale aux LLM.'),
             ('Deux interfaces à distinguer','Le laboratoire web emploie language/command.py et une grammaire limitée. Il n’exécute pas ce cœur neuronal ; la traduction anglaise de trois familles de commandes reste un adaptateur web.')]),
         en=('Ground language in action','A hybrid prototype connects neural candidates, structured parsing and verification before a decision.','Text · structure · decision',[
             ('An existing hybrid prototype','Compact WCE Language Model uses a small character-level neural core and parses clauses, entities and relations. It can construct goals and transformation programs.'),
             ('Verify before deciding','StructuralVerifier and AntiHallucinationPolicy support ACCEPT, VERIFY, REROUTE, ABSTAIN, REJECT and ASK decisions. These form an experimental policy, not a guarantee against hallucination.'),
             ('Training scope','The corpus is artificial and controlled. The prototype does not cover real French generally, and its results do not establish a general alternative to LLMs.'),
             ('Two separate interfaces','The web laboratory uses language/command.py and a limited grammar. It does not run this neural core; English support for three command families is a web adapter.')])),
    dict(key='measurement', route='/architecture/measurement', status='research', sources=('core/measurement.py','core/measurement_world.py'), tests=('tests/test_measurement.py',),
         fr=('Séparer monde et observation','Un état latent, un capteur et une croyance sont des objets différents, reliés par une mesure.','État latent · mesure · croyance',[
             ('Une chaîne explicite','LatentWorldState, Observable, MeasurementContext, MeasurementOperator et MeasurementOutcome distinguent le monde, la grandeur visée, le contexte, l’opération et l’issue enregistrée.'),
             ('Plusieurs mondes compatibles','CompatibleStateSet représente les états compatibles avec une observation. Le module traite bruit, fusion de mesures, entropie et gain d’information pour orienter une prochaine mesure.'),
             ('Référentiels et unités','ReferenceFrame et FramedValue portent le référentiel et les unités ; des erreurs explicites empêchent certaines comparaisons incompatibles. Les transformations 2D sont classiques.'),
             ('Les analogies ont une limite','Les références à la physique guident l’architecture. Le module ne revendique ni mécanique quantique, ni relativité, ni thermodynamique ; les calculs de Shannon sont mathématiques.')]),
         en=('Separate world and observation','A latent state, a sensor and a belief are different objects connected by measurement.','Latent state · measurement · belief',[
             ('An explicit chain','LatentWorldState, Observable, MeasurementContext, MeasurementOperator and MeasurementOutcome distinguish world, target quantity, context, operation and recorded outcome.'),
             ('Multiple compatible worlds','CompatibleStateSet represents states compatible with an observation. Noise, measurement fusion, entropy and information gain help choose the next measurement.'),
             ('Frames and units','ReferenceFrame and FramedValue carry reference frames and units; explicit errors prevent some incompatible comparisons. The 2D transformations are classical.'),
             ('Limits of the analogies','Physics references guide architecture. The module claims neither quantum mechanics, relativity nor thermodynamics; Shannon calculations are mathematical.')])),
    dict(key='sources', route='/architecture/evidence', status='research', sources=('core/source_reliability.py','core/unified_agent.py','core/truth_anchor.py'), tests=('tests/test_sources.py',),
         fr=('Pondérer les preuves','Fiabilité d’une source, identité d’un objet et vérité observée restent des questions distinctes.','Source · identité · ancrage',[
             ('Une fiabilité conditionnée','SourceReliabilityModel estime des distributions Beta hiérarchiques, avec des a priori par type de source et des mises à jour par source. Une observation inconnue ne compte pas comme vraie ou fausse.'),
             ('Relier descriptions et objets','UnifiedAgent combine identité, preuves pondérées et propagation réversible dans un environnement expérimental indépendant. « Unifié » ne signifie pas que tous les modules WCE sont intégrés au site.'),
             ('Vérifier au-delà de sa référence','ObservedTruthAnchor et BoundaryGenerativeCertificate confrontent un programme à des observations indépendantes et à des contre-exemples. Un certificat auto-cohérent peut être faux : l’ancrage vise précisément cette limite.')]),
         en=('Weight the evidence','Source reliability, object identity and observed truth remain distinct questions.','Source · identity · grounding',[
             ('Conditional reliability','SourceReliabilityModel estimates hierarchical Beta distributions with priors by source type and updates by source. Unknown observations count as neither true nor false.'),
             ('Connect descriptions to objects','UnifiedAgent combines identity, weighted evidence and reversible propagation in an independent experimental environment. “Unified” does not mean all WCE modules are integrated into the site.'),
             ('Check beyond its own reference','ObservedTruthAnchor and BoundaryGenerativeCertificate compare a program with independent observations and counterexamples. A self-consistent certificate can be wrong; grounding addresses that limitation.')])),
    dict(key='certificates', route='/architecture/certificates', status='research', sources=('core/generative_certificate.py','core/truth_anchor.py'), tests=('tests/test_generative.py',),
         fr=('Protéger une compétence','Un générateur compact produit des sondes pour vérifier une région de validité, sans conserver tout le jeu initial.','Générateur · sondes · validité',[
             ('Un contrat commun','CertificateGenerator expose des échantillons ordinaires, de frontière, de forte sensibilité, rares et contrefactuels. Des générateurs continus et symboliques implémentent ce contrat.'),
             ('Conserver une région de validité','GenerativeInvariantCertificate associe génération, référence et vérifications. StoredProbeCertificate fournit une autre stratégie avec sondes stockées ; leurs coûts et couvertures doivent être comparés.'),
             ('Ce qui peut échapper au certificat','La couverture dépend du générateur et des invariants testés. Un certificat ne prouve pas que toute entrée future sera correcte ; les contre-exemples indépendants restent nécessaires.')]),
         en=('Protect a skill','A compact generator produces probes to check a validity region without retaining the entire initial dataset.','Generator · probes · validity',[
             ('A common contract','CertificateGenerator exposes ordinary, boundary, high-sensitivity, rare and counterfactual samples. Continuous and symbolic generators implement that contract.'),
             ('Retain a validity region','GenerativeInvariantCertificate combines generation, a reference and checks. StoredProbeCertificate provides a stored-probe alternative; costs and coverage need comparison.'),
             ('What a certificate can miss','Coverage depends on the generator and tested invariants. A certificate does not prove every future input will be correct; independent counterexamples remain necessary.')])),
    dict(key='learning', route='/architecture/learning', status='research', sources=('core/model_state.py','core/plasticity.py','core/safe_plasticity.py','core/stability.py','core/curvature.py','core/continuum.py'), tests=('tests/test_plasticity.py','tests/test_perturbation.py','tests/test_generative.py'),
         fr=('Adapter sans effacer aveuglément','Une mise à jour candidate est évaluée avant consolidation ; les compétences ont un coût et une région de stabilité.','Adaptation · contrôle · consolidation',[
             ('Le modèle est lui-même un état','ModelState représente modules neuronaux, poids lents et rapides et connexions. Ajouter, activer, agrandir, élaguer ou reconnecter un module sont des transformations explicites, tracées et soumises à des préconditions.'),
             ('La mise à jour est une candidate','safe_plasticity sépare les poids rapides de la consolidation. Des contrôles d’invariants peuvent accepter une adaptation ou entraîner un retour arrière sur des tâches expérimentales.'),
             ('Une géométrie locale de stabilité','curvature estime effets de premier et second ordre, produits Hessienne-vecteur et directions de faible rang. Cette approximation locale n’est pas une garantie globale de rétention.'),
             ('Un flux de compétences','ContinuumSystem explore réactivation, dérive, changement abrupt, réutilisation, composition et création de modules. Certificats, fusion et distillation sont étudiés sous budget.'),
             ('Ne pas confondre les bancs','Le banc de plasticité utilise des codes de tâche ; le système continuum étudie un flux sans task_id. Les comparer exige de conserver leurs protocoles et unités propres.')]),
         en=('Adapt without blindly erasing','A candidate update is checked before consolidation; skills have a cost and a stability region.','Adaptation · checks · consolidation',[
             ('The model is itself a state','ModelState represents neural modules, slow and fast weights and connections. Adding, activating, growing, pruning or reconnecting a module are explicit, traced transformations with preconditions.'),
             ('An update is a candidate','safe_plasticity separates fast weights from consolidation. Invariant checks can accept an update or trigger rollback on experimental tasks.'),
             ('Local stability geometry','curvature estimates first- and second-order effects, Hessian-vector products and low-rank directions. This local approximation is not a global retention guarantee.'),
             ('A stream of skills','ContinuumSystem explores reactivation, drift, abrupt change, reuse, composition and module creation. Certificates, merging and distillation are studied under a budget.'),
             ('Keep benchmarks separate','The plasticity benchmark uses task codes; the continuum system explores a stream without task_id. Comparisons must preserve their distinct protocols and units.')])),
    dict(key='discovery', route='/architecture/discovery', status='research', sources=('core/active_structure.py','core/active_world.py','core/discovery.py'), tests=('tests/test_active.py',),
         fr=('Choisir ce qui départage','Des hypothèses concurrentes guident l’expérience suivante plutôt qu’une recherche isolée des observations.','Hypothèses · expérience · révision',[
             ('Une découverte entrelacée','ActiveStructuralDiscovery maintient une postérieure sur des programmes et des classes d’équivalence fonctionnelle. Les observations révisent la structure et orientent son approfondissement.'),
             ('Séparer découverte et évaluation','Les familles nommées sont dans active_world, côté génération du banc. Le module de découverte travaille sur des observations ; cette séparation vise à limiter les informations privilégiées.'),
             ('Observer les limites','Le modèle de bruit est une entrée connue dans ce dispositif. Les hypothèses réfutées de active_summary.json restent visibles : la découverte active n’est pas présentée comme toujours réussie.'),
             ('Découvrir des transformations','discovery.py explore aussi identification de transformations et de lois par expérience active. Ces boucles de recherche ne sont pas disponibles dans l’interface web du simulateur.')]),
         en=('Choose what distinguishes','Competing hypotheses guide the next experiment rather than a search isolated from observations.','Hypotheses · experiment · revision',[
             ('Interleaved discovery','ActiveStructuralDiscovery maintains a posterior over programs and functional equivalence classes. Observations revise structure and guide deeper exploration.'),
             ('Separate discovery from evaluation','Named families are in active_world on the benchmark-generation side. Discovery operates on observations; this separation aims to limit privileged information.'),
             ('Observe the limitations','The noise model is a known input in this setup. Refuted hypotheses in active_summary.json remain visible; active discovery is not presented as always successful.'),
             ('Discover transformations','discovery.py also explores identifying transformations and laws through active experiments. These research loops are not available in the web simulator.')])),
    dict(key='branching', route='/architecture/branching', status='research', sources=('core/branching.py','core/model_state.py'), tests=('tests/test_perturbation.py',),
         fr=('Explorer des états alternatifs','Une branche partage une base et conserve ses propres modifications pour explorer une configuration candidate.','Base · branche · delta',[
             ('Partager avant de modifier','shared_branch crée de nouveaux conteneurs en partageant les tenseurs de la base. Les opérations du modèle réassignent les tenseurs plutôt que de les modifier en place.'),
             ('Des changements locaux','delta_summary décrit les différences avec la base ; IncrementalSnapshot gère des instantanés incrémentaux. Ce mécanisme permet d’examiner des candidats sans confondre leurs modifications avec l’état initial.'),
             ('Une condition technique importante','Le partage suppose le respect de la discipline de réassignation. Une mutation en place non prévue pourrait casser l’isolation ; ce mécanisme n’est pas un service de forks utilisateurs du futur hub.')]),
         en=('Explore alternative states','A branch shares a base and retains its own changes to explore a candidate configuration.','Base · branch · delta',[
             ('Share before modifying','shared_branch creates new containers while sharing base tensors. Model operations reassign tensors instead of modifying them in place.'),
             ('Local changes','delta_summary describes differences from the base; IncrementalSnapshot manages incremental snapshots. This allows candidate inspection without confusing their changes with the initial state.'),
             ('An important condition','Sharing requires reassignment discipline. Unexpected in-place mutation could break isolation; this mechanism is not a user-fork service for the future hub.')])),
]


def render_family_cards(language, families=None):
    from app.i18n import localized_path
    labels = ('Accessible au laboratoire', 'Prototype de recherche', 'Ouvrir le dossier') if language == 'fr' else ('Available in the laboratory', 'Research prototype', 'Read the dossier')
    from app.diagrams import diagram_svg
    pieces=[]
    for index, item in enumerate(FAMILIES if families is None else families,1):
        title, description, chain, _ = item[language]
        status=labels[0 if item['status']=='web' else 1]
        pieces.append(f'<a class="engine-card engine-{item["key"]}" data-mechanism="{item["key"]}" href="{localized_path(item["route"],language)}"><div class="engine-card-top"><span class="eyebrow">WCE</span><span class="maturity maturity-{item["status"]}">{status}</span></div>{diagram_svg(item["key"],language)}<h3>{escape(title)}</h3><p>{escape(description)}</p><div class="mechanism-chain">{escape(chain)}</div><div class="engine-card-source"><code>{escape(item["sources"][0])}</code><span aria-hidden="true">↗</span></div></a>')
    return '<div class="engine-cards">'+''.join(pieces)+'</div>'


def render_atlas(language, compact=False):
    fr=language=='fr'
    title='Dix mécanismes. Une architecture à explorer.' if fr else 'Ten mechanisms. An architecture to explore.'
    intro=('Le dépôt relie des mondes explicites, des transformations composables et des sous-systèmes d’apprentissage. Cette carte décrit les modules existants ; leur intégration varie selon les prototypes.' if fr else 'The repository connects explicit worlds, composable transformations and learning subsystems. This map describes existing modules; integration varies across prototypes.')
    families=[FAMILIES[i] for i in (0,3,4,6,7,8)] if compact else FAMILIES
    if compact:title='Au-delà du simulateur : les mécanismes WCE.' if fr else 'Beyond the simulator: WCE mechanisms.'
    atlas_label='ATLAS D’ARCHITECTURE' if fr else 'ARCHITECTURE ATLAS'
    controls=''
    if not compact:
        labels=('Rechercher un mécanisme','Ex. : mesure, planning, core/branching.py','Aucun mécanisme ne correspond. Essayez un autre terme.','mécanismes affichés') if fr else ('Find a mechanism','E.g. measurement, planning, core/branching.py','No mechanisms match. Try another term.','mechanisms shown')
        controls=f'<div class="atlas-search" hidden><label for="mechanism-search">{labels[0]}</label><input id="mechanism-search" type="search" placeholder="{labels[1]}" aria-controls="mechanism-results" autocomplete="off"><p class="search-count" role="status" aria-live="polite" data-count-label="{labels[3]}"></p><p class="search-empty" hidden>{labels[2]}</p></div>'
    return f'<section class="section engine-atlas"><div class="section-title"><div><p class="eyebrow">WCE / {atlas_label}</p><h2>{title}</h2></div></div><p class="atlas-intro">{intro}</p>'+controls+render_family_cards(language,families).replace('class="engine-cards"','class="engine-cards" id="mechanism-results"')+'</section>'

"""Shared editorial presentations built from the route and research catalogs."""
from html import escape
from app.i18n import localized_path
from app.research_catalog import FAMILIES, render_atlas

SECTION_COPY = {
 '/research': {
  'fr': ('Des résultats, leur protocole, leurs limites.', 'Lire une mesure exige de connaître la tâche, la distribution et le critère de succès. Parcourez les résultats existants, puis les méthodes et les questions ouvertes.'),
  'en': ('Results, protocols and limitations.', 'Reading a measurement requires its task, distribution and success criterion. Explore existing results, then methods and open questions.')},
 '/architecture': {
  'fr': ('Une carte du code, pas une boîte noire.', 'Du monde symbolique aux sous-systèmes neuronaux : chaque dossier présente une fonction, ses mécanismes, son périmètre et ses points d’entrée dans le dépôt.'),
  'en': ('A map of the code, not a black box.', 'From symbolic worlds to neural subsystems: each dossier describes a function, its mechanisms, scope and repository entry points.')},
 '/vision': {
  'fr': ('Trois horizons. Trois niveaux de preuve.', 'Le code actuel, les questions explorées et le futur hub ont des statuts différents. Cette distinction structure notre ambition scientifique.'),
  'en': ('Three horizons. Three levels of evidence.', 'Current code, research questions and the future hub have different statuses. That distinction shapes our scientific ambition.')},
 '/open': {
  'fr': ('Inspecter. Reproduire. Contribuer.', 'Commencer par le code et un environnement local, puis relier chaque contribution à une expérience vérifiable. La plateforme collaborative reste à construire.'),
  'en': ('Inspect. Reproduce. Contribute.', 'Start with code and a local environment, then connect each contribution to a verifiable experiment. The collaborative platform remains to be built.')},
 '/lab': {
  'fr': ('Comprendre avant d’exécuter.', 'Le laboratoire est une fenêtre sur un moteur symbolique précis. Le guide expose la grammaire, le protocole donne une expérience et les limites décrivent ce que la démonstration ne mesure pas.'),
  'en': ('Understand before executing.', 'The laboratory exposes a specific symbolic engine. The guide explains grammar, the protocol provides an experiment, and the limitations describe what the demonstration does not measure.')},
}


def route_description(route, language):
    from app.content import DOSSIERS
    if route in DOSSIERS:return DOSSIERS[route][language][1]
    descriptions={
      '/research/measurement': ('Couverture prédictive sur 96 cas agrégés, résultats par source et incertitude.', 'Predictive coverage across 96 aggregated cases, results by source and uncertainty.'),
      '/research/continual': ('Rétention, coût mémoire et plasticité sur des flux expérimentaux.', 'Retention, memory cost and plasticity on experimental streams.'),
      '/research/discovery': ('Recherche structurelle active, verdicts positifs, réfutés et non concluants.', 'Active structural search, supported, refuted and inconclusive verdicts.'),
      '/architecture': ('Cartographier les mécanismes, leurs interfaces et leur intégration.', 'Map mechanisms, their interfaces and integration.'),
      '/lab': ('Exécuter une commande et observer un monde temporaire.', 'Execute a command and observe a temporary world.'),
      '/research': ('Examiner les artefacts et leur périmètre expérimental.', 'Inspect artifacts and their experimental scope.'),
    }
    item=descriptions.get(route,('Explorer cette rubrique et ses dossiers.', 'Explore this section and its dossiers.'))
    return item[language=='en']


def render_route_cards(routes, language):
    from app.site import page_title
    from app.content import DOSSIERS
    cards=[]
    for i,route in enumerate(routes,1):
        record=DOSSIERS.get(route)
        status=record[language][2] if record else (('Résultats documentés' if language=='fr' else 'Documented results') if route in ('/research/measurement','/research/continual','/research/discovery') else ('Rubrique' if language=='fr' else 'Section'))
        cards.append(f'<a class="section-card" href="{localized_path(route,language)}"><div class="section-card-top"><!-- card {i} --><span class="maturity">{escape(status)}</span></div><h3>{escape(page_title(route,language))}</h3><p>{escape(route_description(route,language))}</p><span class="section-card-link">'+('Explorer le dossier' if language=='fr' else 'Explore the dossier')+' <span aria-hidden="true">↗</span></span></a>')
    return '<div class="section-cards">'+''.join(cards)+'</div>'


def render_section_directory(route,language):
    from app.site import NAVIGATION
    group=next((g for g in NAVIGATION if g[0]==route),None)
    if not group:return ''
    title,intro=SECTION_COPY[route][language]
    return f'<section class="section section-directory"><p class="eyebrow">'+('DANS CETTE RUBRIQUE' if language=='fr' else 'IN THIS SECTION')+f'</p><h2>{title}</h2><p class="atlas-intro">{intro}</p>'+render_route_cards(group[3][1:],language)+'</section>'


def render_architecture(language):
    fr=language=='fr'
    title='Des primitives aux systèmes.<br><em>Les mécanismes à découvert.</em>' if fr else 'From primitives to systems.<br><em>Mechanisms in the open.</em>'
    intro=('WCE explore la complexité par composition : des transformations explicites, des invariants, des observations et des mécanismes d’adaptation. Le dépôt contient plusieurs prototypes qui partagent cette direction, avec des niveaux d’intégration différents.' if fr else 'WCE explores complexity through composition: explicit transformations, invariants, observations and adaptation mechanisms. The repository contains several prototypes that share this direction, with different integration levels.')
    steps=[('Commande','language/command.py'),('Action','core/executor.py'),('Monde','core/world.py'),('Simulation','physics/engine.py'),('Réponse','language/answer.py')]
    if not fr:steps=[('Command',p) if n=='Commande' else ('World',p) if n=='Monde' else ('Response',p) if n=='Réponse' else (n,p) for n,p in steps]
    flow=''.join(f'<li><!-- step {i} --><h3>{n}</h3><code>{p}</code></li>' for i,(n,p) in enumerate(steps,1))
    return f'<section class="page-head"><p class="eyebrow">ARCHITECTURE</p><h1>{title}</h1><p class="lead">{intro}</p></section><section class="section runtime-section"><div class="section-title"><h2>'+('Le parcours servi aujourd’hui.' if fr else 'The pathway served today.')+'</h2><span class="maturity maturity-web">'+('LABORATOIRE WEB' if fr else 'WEB LABORATORY')+'</span></div><ol class="runtime-flow">'+flow+'</ol><p class="note">'+('Le moteur symbolique relie analyse, action, état et simulation. Les sous-systèmes de recherche présentés ci-dessous ont leurs propres protocoles ; ils ne constituent pas encore un modèle intégré.' if fr else 'The symbolic engine connects parsing, actions, state and simulation. The research subsystems below have their own protocols; they do not yet constitute an integrated model.')+'</p></section>'+render_atlas(language)+render_integration(language)


def render_integration(language):
    fr=language=='fr'
    states=[
      ('Aujourd’hui','Un simulateur web et des prototypes de recherche dans le dépôt.','Today','A web simulator and research prototypes in the repository.'),
      ('À explorer','Des interfaces communes, des objectifs et des évaluations entre sous-systèmes.','Being explored','Common interfaces, goals and evaluations across subsystems.'),
      ('Ambition','Un hub ouvert : architectures, expériences reproductibles et contributions.','Ambition','An open hub: architectures, reproducible experiments and contributions.')]
    body=''.join(f'<article><!-- section {i} --><h3>{item[0 if fr else 2]}</h3><p>{item[1 if fr else 3]}</p></article>' for i,item in enumerate(states,1))
    return '<section class="section integration-panel"><h2>'+('Relier les briques reste un travail de recherche.' if fr else 'Connecting the components remains a research task.')+'</h2><div class="integration-grid">'+body+'</div><a class="text-link" href="'+localized_path('/vision',language)+'">'+('Voir les trois horizons →' if fr else 'Explore the three horizons →')+'</a></section>'


def render_inventory(language):
    import json
    from pathlib import Path
    records=json.loads(Path(__file__).with_name('backend_inventory.json').read_text())
    fr=language=='fr'
    labels={'core':('Noyau et recherche','Core and research'),'language':('Analyse et réponses','Parsing and answers'),'physics':('Simulation physique','Physical simulation'),'logic':('Logique','Logic'),'app':('Interfaces et sessions','Interfaces and sessions')}
    groups=[]
    for group in labels:
        items=[r for r in records if r['group']==group]
        if not items:continue
        rows=[]
        for item in items:
            symbols=item['classes']+item['functions']
            tags=''.join('<code>'+escape(symbol)+'</code>' for symbol in symbols[:5])
            if len(symbols)>5:
                more=''.join('<code>'+escape(symbol)+'</code>' for symbol in symbols[5:])
                tags+=f'<details class="inventory-symbols"><summary>+{len(symbols)-5}</summary><div>{more}</div></details>'
            rows.append('<li data-source-module><code class="inventory-path">'+escape(item['path'])+'</code><div>'+tags+'</div></li>')
        groups.append('<details class="inventory-group"><summary><span>'+labels[group][not fr]+f'</span><span>{len(items)} modules +</span></summary><ul>'+''.join(rows)+'</ul></details>')
    title='Le dépôt, au-delà des dix dossiers.' if fr else 'The repository beyond the ten dossiers.'
    intro=('Cet inventaire couvre les fichiers Python des dossiers core, language, physics, logic et les interfaces moteur de app. Les noms viennent d’une analyse statique du code ; ils ne certifient ni la maturité ni l’intégration des modules. Les outils éditoriaux et fichiers d’initialisation sont exclus.' if fr else 'This inventory covers Python files in core, language, physics, logic and the engine interfaces in app. Names come from static code analysis; they certify neither maturity nor integration. Editorial utilities and package initialisers are excluded.')
    search_label='Rechercher un module ou un symbole' if fr else 'Find a module or symbol'
    placeholder='Ex. : planner, World, core/measurement.py' if fr else 'E.g. planner, World, core/measurement.py'
    count_label='modules affichés' if fr else 'modules shown'
    empty='Aucun module ne correspond. Essayez un autre nom ou effacez la recherche.' if fr else 'No modules match. Try another name or clear the search.'
    controls=f'<div class="inventory-search" hidden><label for="source-search">{search_label}</label><input id="source-search" type="search" placeholder="{placeholder}" aria-controls="source-results" autocomplete="off"><p class="search-count" role="status" data-count-label="{count_label}"></p><p class="search-empty" hidden>{empty}</p></div>'
    return f'<section class="section backend-inventory"><p class="eyebrow">SOURCE MAP / {len(records)} MODULES</p><h2>{title}</h2><p class="atlas-intro">{intro}</p>'+controls+'<div id="source-results">'+''.join(groups)+'</div></section>'

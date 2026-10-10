"""Public bilingual reading aids grounded in the scientific engine."""
from html import escape


def introduction(language):
    fr = language == 'fr'
    cards = [
        ('QUESTION', 'Comment décrire ce qui change ?', 'Un état décrit une situation. Une transformation la modifie. WCE étudie comment rendre ces opérations explicites, les combiner et vérifier ce qu’elles préservent.', '/architecture/primitives', 'Comprendre les mécanismes',
         'QUESTION', 'How can we describe change?', 'A state describes a situation. A transformation changes it. WCE studies how to make these operations explicit, combine them and check what they preserve.', '/architecture/primitives', 'Understand the mechanisms'),
        ('PREUVE', 'Que peut-on vérifier aujourd’hui ?', 'Un moteur symbolique fait évoluer un monde réduit. Des expériences distinctes évaluent observation, rétention et découverte de structures. Elles ne forment pas encore un modèle unique.', '/research', 'Examiner les résultats',
         'EVIDENCE', 'What can we verify today?', 'A symbolic engine changes a small world. Separate experiments evaluate observation, retention and structure discovery. They do not yet form a single model.', '/research', 'Examine the results'),
        ('AMBITION', 'Que voulons-nous rendre possible ?', 'Permettre à chacun d’inspecter une architecture, de reproduire ses expériences et d’en proposer une autre. Le hub ouvert et l’entraînement unifié restent à construire.', '/vision', 'Lire les objectifs et les limites',
         'AMBITION', 'What do we want to enable?', 'Enable anyone to inspect an architecture, reproduce its experiments and propose another one. The open hub and unified training remain goals to build towards.', '/vision', 'Read the goals and limitations')]
    body = ''
    for item in cards:
        label, title, text, route, link = item[:5] if fr else item[5:]
        body += f'<article><p class="eyebrow">{label}</p><h3>{title}</h3><p>{text}</p><a class="reading-link" href="{route}">{link} <span aria-hidden="true">↗</span></a></article>'
    return '<section class="section understanding"><div class="section-title"><h2>'+('Comprendre WCE.' if fr else 'Understanding WCE.')+'</h2><span class="eyebrow">'+('QUESTION · PREUVE · AMBITION' if fr else 'QUESTION · EVIDENCE · AMBITION')+'</span></div><div class="understanding-grid">'+body+'</div></section>'


def glossary(language):
    fr = language == 'fr'
    items = [
        ('Primitive', 'Une brique élémentaire utilisée pour construire une opération ou une structure. Le choix des briques dépend du prototype.', 'Primitive', 'A basic building block used to construct an operation or structure. The choice of blocks depends on the prototype.'),
        ('Configuration', 'L’état décrit à un instant donné : objets, propriétés, relations ou structure d’un modèle.', 'Configuration', 'The state described at a given moment: objects, properties, relationships or a model’s structure.'),
        ('Transformation', 'Une opération qui produit une nouvelle configuration, avec des conditions et des changements déclarés.', 'Transformation', 'An operation that produces a new configuration, with declared conditions and changes.'),
        ('Invariant', 'Une propriété qu’une opération déclare préserver. Cette déclaration doit être vérifiée dans son domaine de validité.', 'Invariant', 'A property an operation declares it preserves. This declaration must be checked within its domain of validity.'),
        ('Composition', 'L’assemblage de plusieurs opérations en une séquence réutilisable. Leur compatibilité doit être vérifiée.', 'Composition', 'Combining operations into a reusable sequence. Their compatibility must be checked.'),
        ('Apprentissage structurel', 'Faire évoluer une structure, pas seulement ses valeurs numériques. WCE explore cette idée dans des expériences spécifiques.', 'Structural learning', 'Changing a structure, rather than only its numerical values. WCE explores this idea in specific experiments.')]
    body = ''.join(f'<details><summary>{escape(a if fr else c)}<span aria-hidden="true">+</span></summary><p>{escape(b if fr else d)}</p></details>' for a,b,c,d in items)
    return '<section class="section vocabulary"><p class="eyebrow">'+('REPÈRES' if fr else 'KEY IDEAS')+'</p><h2>'+('Six mots pour lire l’architecture.' if fr else 'Six terms for reading the architecture.')+'</h2><p class="atlas-intro">'+('Des définitions accessibles ; les dossiers précisent ensuite leurs implémentations et leurs limites.' if fr else 'Accessible definitions; the dossiers then explain their implementations and limitations.')+'</p><div class="vocabulary-grid">'+body+'</div></section>'


QUESTIONS = {
 '/research/measurement': (
  ('Peut-on exprimer l’incertitude sans la confondre avec la vérité ?', 'L’artefact distingue couverture prédictive et couverture crédible, par type de source. Une bonne couverture sur ces observations ne garantit pas la fiabilité dans un autre contexte.'),
  ('Can we express uncertainty without confusing it with truth?', 'The artifact distinguishes predictive and credible coverage by source type. Good coverage on these observations does not guarantee reliability in another context.')),
 '/research/continual': (
  ('Comment apprendre de nouvelles tâches sans perdre les précédentes ?', 'Les stratégies sont comparées sur des flux expérimentaux : performance, rétention et coûts. Les résultats dépendent des tâches, des graines et du budget ; ils ne démontrent pas un apprentissage général sans oubli.'),
  ('How can we learn new tasks without losing earlier ones?', 'Strategies are compared on experimental streams: performance, retention and costs. Results depend on tasks, seeds and budget; they do not establish general learning without forgetting.')),
 '/research/discovery': (
  ('Quelle expérience choisir pour découvrir une structure inconnue ?', 'Le banc mesure des stratégies de découverte dans des mondes générés pour l’évaluation. Les verdicts soutenus, réfutés et non concluants restent visibles ; ils ne prouvent pas la découverte autonome de lois du monde réel.'),
  ('Which experiment should we choose to discover an unknown structure?', 'The benchmark measures discovery strategies in worlds generated for evaluation. Supported, refuted and inconclusive verdicts remain visible; they do not prove autonomous discovery of real-world laws.')),
}


def research_question(route, language):
    title, text = QUESTIONS[route][language == 'en']
    return '<section class="section research-question"><p class="eyebrow">'+('LA QUESTION ÉTUDIÉE' if language == 'fr' else 'THE RESEARCH QUESTION')+f'</p><h2>{title}</h2><p>{text}</p></section>'

# Curated links: a related experiment is not direct validation of a whole family.
EVIDENCE_ROUTES = {
 'measurement': '/research/measurement', 'sources': '/research/measurement',
 'learning': '/research/continual', 'certificates': '/research/continual',
 'discovery': '/research/discovery',
}


def mechanism_scope(route, language):
    from app.research_catalog import FAMILIES
    family = next((item for item in FAMILIES if item['route'] == route), None)
    if family is None:
        return ''
    fr = language == 'fr'
    web = family['status'] == 'web'
    states = [
        ('Code identifié', 'Les fichiers cités ci-dessous décrivent les objets et les opérations de ce mécanisme.', 'Identified code', 'The files cited below describe this mechanism’s objects and operations.'),
        ('Vérification ciblée', 'Les tests cités vérifient des comportements définis. Leur présence ne prouve pas une capacité générale.', 'Targeted verification', 'The cited tests check defined behaviours. Their presence does not prove a general capability.'),
        ('Intégration', 'Ce mécanisme est utilisé dans le monde réduit du laboratoire.' if web else 'Ce mécanisme appartient à un prototype de recherche. Son intégration à un modèle commun reste à évaluer.', 'Integration', 'This mechanism is used in the laboratory’s small world.' if web else 'This mechanism belongs to a research prototype. Integration into a shared model remains to be evaluated.')]
    body = ''.join(f'<article><!-- section {i} --><h3>{escape(a if fr else c)}</h3><p>{escape(b if fr else d)}</p></article>' for i,(a,b,c,d) in enumerate(states,1))
    evidence = EVIDENCE_ROUTES.get(family['key'])
    link = ('<a class="reading-link" href="'+evidence+'">'+('Expérience associée — portée à lire dans son protocole' if fr else 'Related experiment — read its protocol for scope')+' <span aria-hidden="true">↗</span></a>') if evidence else ('<p class="note">'+('Aucun dossier de résultats dédié à cette famille n’est publié ici. Consulter le code et les tests ne remplace pas un benchmark.' if fr else 'No dedicated results dossier for this family is published here. Inspecting code and tests does not replace a benchmark.')+'</p>')
    return '<section class="section mechanism-scope"><p class="eyebrow">'+('COMMENT LIRE CE DOSSIER' if fr else 'HOW TO READ THIS DOSSIER')+'</p><h2>'+('Du mécanisme à la preuve.' if fr else 'From mechanism to evidence.')+'</h2><div class="scope-grid">'+body+'</div>'+link+'</section>'


def public_release_state(language):
    fr=language=='fr'
    items=[
      ('Consultable ici','Les dossiers, trois artefacts JSON et les protocoles scientifiques cités peuvent être lus sur ce site.','Available here','The dossiers, three JSON artifacts and cited scientific protocols can be read on this site.'),
      ('Dépôt canonique','Le code scientifique courant est publié sur github.com/wce-research/wce sous licence Apache-2.0. Les sources et documents scientifiques sont en anglais.','Canonical repository','Current scientific source is published at github.com/wce-research/wce under Apache-2.0. Scientific source and documentation are in English.'),
      ('Objectif de collaboration','Contributions, forks et comparaisons reproductibles passent par GitHub et CONTRIBUTING.md. Le hub collaboratif sur ce site reste une vision.','Collaboration goal','Contributions, forks and reproducible comparisons use GitHub and CONTRIBUTING.md. A collaborative hub on this website remains a vision.')]
    cards=''.join(f'<article><!-- card {i} --><h3>{a if fr else c}</h3><p>{b if fr else d}</p></article>' for i,(a,b,c,d) in enumerate(items,1))
    return '<section class="section release-state"><p class="eyebrow">'+('ÉTAT DE L’OUVERTURE' if fr else 'OPEN RESEARCH STATUS')+'</p><h2>'+('Ce qui est accessible. Ce qui reste à ouvrir.' if fr else 'What is accessible. What remains to be opened.')+'</h2><div class="release-grid">'+cards+'</div></section>'


def contribution_brief(language):
    fr=language=='fr'
    text=('Question étudiée :\nMécanisme modifié :\nVersion du code et dépendances :\nDonnées et graines :\nBaseline et budget :\nCritères annoncés avant exécution :\nCommande de reproduction :\nRésultats et empreintes des artefacts :\nLimites et résultats négatifs :'
          if fr else 'Research question:\nModified mechanism:\nCode version and dependencies:\nData and seeds:\nBaseline and budget:\nCriteria declared before execution:\nReproduction command:\nResults and artifact hashes:\nLimitations and negative results:')
    return '<section class="section contribution-brief"><p class="eyebrow">'+('PRÉPARER UNE CONTRIBUTION' if fr else 'PREPARE A CONTRIBUTION')+'</p><h2>'+('Une idée doit devenir une expérience vérifiable.' if fr else 'An idea needs an experiment others can check.')+'</h2><p class="atlas-intro">'+('Ce canevas aide à documenter une proposition scientifique. Les propositions suivent CONTRIBUTING.md dans le dépôt GitHub canonique ; cette page n’est pas un formulaire d’envoi.' if fr else 'This outline helps document a scientific proposal. Proposals follow CONTRIBUTING.md in the canonical GitHub repository; this page is not a submission form.')+'</p><pre><code>'+escape(text)+'</code></pre></section>'


def research_preview(language):
    from app.evidence import artifact
    from collections import Counter
    fr=language=='fr'
    coverage,_=artifact('predictive_coverage.json')
    continuum,_=artifact('continuum_summary.json')
    discovery,_=artifact('active_summary.json')
    counts=Counter(record['verdict'] for record in discovery['hypotheses'].values())
    studies=[
      ('OBSERVATION','Exprimer l’incertitude','La couverture varie selon le type de source. Lire les intervalles et les limites avant de conclure.',str(coverage['all']['n']),'cas agrégés','/research/measurement',
       'OBSERVATION','Express uncertainty','Coverage varies by source type. Read the intervals and limitations before drawing conclusions.','aggregated cases'),
      ('APPRENTISSAGE','Apprendre dans la durée','Comparer acquisition, rétention et coûts sur un flux expérimental, sans les réduire à un classement unique.',str(len(continuum['methods'])),'méthodes comparées','/research/continual',
       'LEARNING','Learn over time','Compare acquisition, retention and costs on an experimental stream without reducing them to a single ranking.','compared methods'),
      ('DÉCOUVERTE','Mettre les hypothèses à l’épreuve','Les résultats soutenus, réfutés et non concluants font partie du même dossier.',f'{counts["SOUTENUE"]} / {counts["RÉFUTÉE"]} / {counts["NON CONCLUANTE"]}','soutenues / réfutées / non concluantes','/research/discovery',
       'DISCOVERY','Put hypotheses to the test','Supported, refuted and inconclusive results belong in the same record.','supported / refuted / inconclusive')]
    cards=''
    for label,title,text,value,unit,route,en_label,en_title,en_text,en_unit in studies:
        if not fr:label,title,text,unit=en_label,en_title,en_text,en_unit
        cards+=f'<a class="study-card" href="{route}"><p class="eyebrow">{label}</p><h3>{title}</h3><p>{text}</p><div class="study-value"><strong>{value}</strong><span>{unit}</span></div><span class="reading-link">'+('Lire le protocole et les résultats' if fr else 'Read the protocol and results')+' <span aria-hidden="true">↗</span></span></a>'
    return '<section class="section research-preview"><div class="section-title"><div><p class="eyebrow">'+('TRACES EXPÉRIMENTALES' if fr else 'EXPERIMENTAL RECORDS')+'</p><h2>'+('Trois questions. Des résultats inspectables.' if fr else 'Three questions. Inspectable results.')+'</h2></div></div><div class="study-grid">'+cards+'</div><p class="note">'+('Les effectifs viennent des artefacts publiés. Ils décrivent ces expériences, pas un niveau d’intelligence ni une supériorité générale.' if fr else 'Counts come from the published artifacts. They describe these experiments, not a level of intelligence or general superiority.')+'</p><a class="button" href="/architecture">'+('Explorer les dix mécanismes WCE ↗' if fr else 'Explore WCE’s ten mechanisms ↗')+'</a></section>'

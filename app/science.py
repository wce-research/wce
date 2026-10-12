"""Public science views: read allowlisted artifacts, never import research runtimes."""
from functools import lru_cache
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re

from app.evidence import number
from app.public_identity import GITHUB_REPO
from app.science_catalog import MILESTONES, PUBLIC_SUMMARIES, LATEST_VERDICT, SCIENCE_ROUTES

ROOT = Path(__file__).resolve().parents[1]


def text(pair, language):
    return pair[language == 'fr']


def source_link(path, label=None):
    kind = 'tree' if path == 'tests' else 'blob'
    return f'<a href="{GITHUB_REPO}/{kind}/main/{escape(path, quote=True)}">{escape(label or path)} ↗</a>'


@lru_cache(maxsize=len(PUBLIC_SUMMARIES))
def summary(name):
    path = 'results/' + name
    if path not in PUBLIC_SUMMARIES:
        raise ValueError('Artifact is not in the public science allowlist')
    raw = (ROOT / path).read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def provenance(name, language):
    _, digest = summary(name)
    label = text(('Source JSON and full fingerprint', 'JSON source et empreinte complète'), language)
    return f'<details class="artifact-hash science-provenance"><summary>{label}</summary><p>{source_link("results/" + name)}</p><a href="/research/{name}">{escape(name)}</a><code>{digest}</code></details>'


def head(route, language, intro):
    return '<section class="page-head"><p class="eyebrow">WCE / OPEN RESEARCH</p><h1>' + text(SCIENCE_ROUTES[route], language) + '</h1><p class="lead">' + escape(text(intro, language)) + '</p></section>'


def status(language, compact=False):
    intro = text(('Current scientific verdict', 'Verdict scientifique actuel'), language)
    explanation = text((
        'Small synthetic worlds, tested components, an unresolved full chain. Better atom estimates have not produced reliable deep reconstruction without an oracle.',
        'Petits mondes synthétiques, composants testés, chaîne complète non résolue. De meilleures estimations des atomes ne produisent pas encore une reconstruction profonde fiable sans oracle.'), language)
    result = f'<section class="section scientific-status"><p class="eyebrow">{intro}</p><h2>{LATEST_VERDICT}</h2><p>{explanation}</p><a class="text-link" href="/research/high-precision">' + text(('Read the result and its limits →', 'Lire le résultat et ses limites →'), language) + '</a>'
    if not compact:
        states = [
            ('IMPLEMENTED / CURRENT', ('Symbolic world, transformation objects, tested composition and fitting code. Tested implementation is not confirmation of its scientific hypothesis.', 'Monde symbolique, objets de transformation, code de composition et d’ajustement testé. Une implémentation testée ne confirme pas son hypothèse scientifique.')),
            ('EXPERIMENTAL', ('Plasticity, language, belief, discovery and estimator prototypes are separately evaluated; they are not one integrated trainable model.', 'Les prototypes de plasticité, langage, croyance, découverte et estimation sont évalués séparément ; ils ne forment pas un modèle entraînable intégré.')),
            ('BLOCKED / OPEN PROBLEM', ('Complete primitive recovery and joint parameter precision under composition and occlusion remain inadequate.', 'Récupération complète des primitives et précision conjointe sous composition et occlusion restent insuffisantes.')),
            ('FUTURE / RESEARCH DIRECTION', ('Common training interfaces, stronger estimators and real-scene evaluation require new implementations and protocols.', 'Interfaces d’entraînement communes, meilleurs estimateurs et évaluation de scènes réelles nécessitent de nouvelles implémentations et protocoles.')),
            ('LONG-TERM VISION', ('A unified inspectable architecture and an open research hub are ambitions; WCE is not a demonstrated AGI, LLM replacement or autonomous scientist.', 'Une architecture unifiée inspectable et un hub ouvert sont des ambitions ; WCE n’est ni une AGI démontrée, ni un remplacement des LLM, ni un scientifique autonome opérationnel.')),
        ]
        result += '<div class="science-status-grid">' + ''.join('<article><h3>' + key + '</h3><p>' + text(copy, language) + '</p></article>' for key, copy in states) + '</div>'
    return result + '</section>'


def programme(language):
    result = head('/research/programme', language, (
        'Follow how each hypothesis changed the next experiment. Failed and inconclusive experiments remain part of the record.',
        'Suivre comment chaque hypothèse a changé l’expérience suivante. Échecs et expériences non concluantes restent dans le dossier.'))
    result += status(language)
    result += '<section class="section"><h2>' + text(('Choose a reading path', 'Choisir un parcours de lecture'), language) + '</h2><div class="science-status-grid">'
    paths = [
        ('/architecture/research-map', ('Understand the mechanisms', 'Comprendre les mécanismes'), ('Start with world, observation and reversible transformations; then explore the code hierarchy.', 'Commencer par monde, observation et transformations réversibles ; puis explorer la hiérarchie du code.')),
        ('/research/deep-composition', ('Evaluate the evidence', 'Évaluer les preuves'), ('Compare exact-atom composition with the limits of discovered atoms; read intervals and protocols.', 'Comparer composition avec atomes exacts et limites des atomes découverts ; lire intervalles et protocoles.')),
        ('/open/reproduce', ('Reproduce or contribute', 'Reproduire ou contribuer'), ('Inspect reports, budgets, Makefile targets and contribution rules before running an experiment.', 'Inspecter rapports, budgets, cibles Makefile et règles de contribution avant une expérience.')),
    ]
    result += ''.join(f'<article><h3><a href="{route}">{text(title,language)} →</a></h3><p>{text(copy,language)}</p></article>' for route,title,copy in paths) + '</div></section>'
    result += '<section class="section"><p class="eyebrow">RESEARCH EVOLUTION</p><h2>' + text(('A scientific progression, including what failed', 'Une progression scientifique, y compris les échecs'), language) + '</h2><p>' + text(('Ordered by research dependency, not an invented calendar. Expand a milestone to follow hypothesis → experiment → result → consequence. Editorial summaries do not replace report verdicts.', 'Ordre des dépendances scientifiques, sans calendrier inventé. Déplier un jalon pour suivre hypothèse → expérience → résultat → conséquence. Les résumés éditoriaux ne remplacent pas les verdicts des rapports.'), language) + '</p><ol class="science-timeline">'
    for report, artifact, verdict, title, hypothesis, outcome, next_question in MILESTONES:
        report_path = 'docs/rapport_' + report + '.md'
        report_text = (ROOT/report_path).read_text()
        verdict_kind = text(('Report verdict', 'Verdict du rapport'),language) if verdict in report_text else text(('Editorial synthesis', 'Synthèse éditoriale'),language)
        labels = text((('Hypothesis','Experiment','Result and consequence','Next question'), ('Hypothèse','Expérience','Résultat et conséquence','Question suivante')), language)
        result += f'<li><details data-milestone="{report}"><summary><span>{text(title,language)}</span><strong class="science-verdict">{verdict}<small>{verdict_kind}</small></strong></summary><dl><dt>{labels[0]}</dt><dd>{text(hypothesis,language)}</dd><dt>{labels[1]}</dt><dd>{source_link(report_path, text(("Protocol, arms, budgets and report", "Protocole, bras, budgets et rapport"), language))} · <a href="/research/{artifact}">JSON ↗</a></dd><dt>{labels[2]}</dt><dd>{text(outcome,language)}</dd><dt>{labels[3]}</dt><dd>{text(next_question,language)}</dd></dl>{provenance(artifact,language)}</details></li>'
    return result + '</ol><p class="note">' + text(('The latest deep milestones use development data, not a global confirmatory evaluation. Nothing is enabled by default in the frozen reference pipeline.', 'Les derniers jalons profonds utilisent des données de développement, pas une évaluation confirmatoire globale. Rien n’est activé par défaut dans la référence figée.'), language) + '</p></section>'


def numeric_cell(name, pointer, value, language):
    # The pointer is a JSON array of literal keys; slash-containing keys stay unambiguous.
    return f'<td data-science-artifact="{name}" data-science-path="{escape(json.dumps(pointer),quote=True)}" data-science-value="{value}">{number(value,language)}</td>'


def bar(value):
    return f'<svg viewBox="0 0 100 10" class="science-bar" aria-hidden="true"><path d="M0 5H100" class="coverage-track"/><path d="M0 5H{value*100:.4f}" class="coverage-predictive"/></svg>'


def paired_table(name, section, keys, metrics, language):
    data, _ = summary(name)
    rows = []
    for key in keys:
        for metric, label in metrics:
            record = data[section][key][metric]
            cells = numeric_cell(name, [section,key,metric,'diff'], record['diff'], language)
            interval = ' / '.join(number(x,language) for x in record['ci95'])
            cells += f'<td data-science-artifact="{name}" data-science-path="{escape(json.dumps([section,key,metric,"ci95"]),quote=True)}" data-science-value="{escape(json.dumps(record["ci95"]),quote=True)}">[{interval}]</td>'
            cells += numeric_cell(name, [section,key,metric,'n'], record['n'], language)
            rows.append(f'<tr><th scope="row">{escape(key)} · {text(label,language)}</th>{cells}</tr>')
    return '<div class="table-scroll"><table><caption>' + text(('Paired differences; 95% bootstrap intervals', 'Différences appariées ; intervalles bootstrap à 95 %'),language) + '</caption><thead><tr><th scope="col">' + text(('Comparison / metric','Comparaison / métrique'),language) + '</th><th scope="col">Δ</th><th scope="col">95% CI</th><th scope="col">n</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def composition(language):
    name = 'deep_composition_oracle_summary.json'; data, _ = summary(name)
    result = head('/research/deep-composition', language, ('Composition improves dramatically when true atoms and true parameters are provided. This controlled result does not solve discovery.', 'La composition progresse fortement lorsque les atomes et paramètres vrais sont fournis. Ce résultat contrôlé ne résout pas la découverte.'))
    result += '<section class="section"><p class="science-verdict">DEEP COMPOSITION READY FOR END-TO-END VALIDATION</p><h2>' + text(('Exact atoms supplied — development experiment', 'Atomes exacts fournis — expérience de développement'),language) + '</h2><p>' + text(('Both arms receive the same true atoms. Functional recovery means agreement ≥ 0.99 with the target function, not recovery of the unique original tree. Classes merge expressions that behave alike on observed cells; fewer redundant expressions make deeper search feasible.', 'Les deux bras reçoivent les mêmes atomes vrais. La récupération fonctionnelle signifie un accord ≥ 0,99 avec la fonction cible, pas la récupération d’un arbre original unique. Les classes regroupent les expressions au comportement identique sur les cellules observées ; cette réduction des redondances permet une recherche plus profonde.'),language) + '</p>'
    result += '<div class="table-scroll"><table><caption>' + text(('Share of tasks with functional recovery ≥ 0.99 · scale 0–1', 'Part des tâches avec récupération fonctionnelle ≥ 0,99 · échelle 0–1'),language) + '</caption><thead><tr><th scope="col">' + text(('Atoms','Atomes'),language) + '</th><th scope="col">ref_b150</th><th scope="col">' + text(('Equivalence classes (errors)', 'Classes d’équivalence (erreurs)'),language) + '</th><th scope="col">n</th></tr></thead><tbody>'
    for depth in range(1,7):
        reference = data['reference'][f'd{depth}/ref_b150']; selected = data['selectors'][f'd{depth}/errors']
        result += f'<tr><th scope="row">{depth}</th>' + numeric_cell(name,['reference',f'd{depth}/ref_b150','functional'],reference['functional'],language) + numeric_cell(name,['selectors',f'd{depth}/errors','functional'],selected['functional'],language).replace('</td>',bar(selected['functional'])+'</td>') + numeric_cell(name,['reference',f'd{depth}/ref_b150','n'],reference['n'],language) + '</tr>'
    result += '</tbody></table></div>' + paired_table(name,'paired',['all: dp_errors − ref_b150'],[('functional',('Functional recovery','Récupération fonctionnelle'))],language)
    result += '<p class="note">' + text(('Protocol: up to six exact supplied atoms, development keys, one task seed per key. Positive intervals support this comparison within its protocol; no global confirmatory result or live sealed evaluation follows.', 'Protocole : jusqu’à six atomes exacts fournis, clés de développement, une graine par clé. Les intervalles positifs soutiennent cette comparaison dans son protocole ; aucune confirmation globale ni évaluation scellée active n’en découle.'),language) + '</p><p>' + source_link('docs/rapport_deep_composition_oracle_atoms.md') + '</p>' + provenance(name,language) + '</section>'
    result += '<section class="section"><h2>' + text(('Why this does not solve end-to-end reconstruction', 'Pourquoi cela ne résout pas la reconstruction de bout en bout'),language) + '</h2><p>' + text(('The controlled experiment bypasses the hardest upstream problem: discovering every primitive with jointly accurate parameters. End-to-end, latent-set and High-Precision experiments subsequently fail their preregistered criteria. Missing atoms cannot be repaired by composing the wrong pool more efficiently.', 'L’expérience contrôlée contourne le problème amont le plus difficile : découvrir chaque primitive avec des paramètres conjointement précis. Les expériences de bout en bout, d’ensembles latents et High-Precision échouent ensuite aux critères préenregistrés. Composer plus efficacement un mauvais ensemble ne répare pas les atomes manquants.'),language) + '</p><a class="text-link" href="/research/high-precision">' + text(('Inspect the current bottleneck →','Inspecter le blocage actuel →'),language) + '</a></section>'
    return result


def high_precision(language):
    name = 'deep_high_precision_summary.json'; data, _ = summary(name)
    result = head('/research/high-precision',language,('Milestone completed. Specialized geometric estimators improve individual estimates, but reliable deep reconstruction remains blocked.', 'Jalon terminé. Les estimateurs géométriques spécialisés améliorent les estimations individuelles, mais la reconstruction profonde fiable reste bloquée.'))
    result += status(language,True)
    result += '<section class="section"><h2>' + text(('Primary experiment: better averages, fewer high-accuracy successes', 'Expérience principale : meilleures moyennes, moins de réussites à haute exactitude'),language) + '</h2><p>' + text(('A is the frozen upstream, B is flat_scene_k3, C adds specialized high-precision estimators. They share the strong downstream. Primary standard/rich regimes use development tasks and one seed per key. Agreement is mean pointwise correctness; final ≥ 0.99 is the share of tasks meeting that threshold. They answer different questions.', 'A utilise l’amont figé, B flat_scene_k3, C ajoute les estimateurs spécialisés haute précision. Ils partagent le même aval renforcé. Les régimes principaux standard/rich utilisent des tâches de développement et une graine par clé. L’accord est l’exactitude moyenne point par point ; final ≥ 0,99 est la part des tâches atteignant ce seuil. Ce sont des questions différentes.'),language) + '</p>'
    columns = [('accord',('Mean agreement','Accord moyen')),('functional',('Final ≥ 0.99','Final ≥ 0,99')),('pool_recall_functional',('Atom recall','Rappel des atomes')),('pool_all_functional',('Complete pool','Ensemble complet')),('seconds',('Seconds/task','Secondes/tâche'))]
    result += '<div class="table-scroll"><table><caption>' + text(('Primary regimes; observed values from the public summary','Régimes principaux ; valeurs observées du résumé public'),language) + '</caption><thead><tr><th scope="col">' + text(('Arm','Bras'),language) + '</th>' + ''.join('<th scope="col">'+text(label,language)+'</th>' for _,label in columns) + '<th scope="col">n</th></tr></thead><tbody>'
    for arm in ('A','B','C'):
        key='primary/all/'+arm; record=data['chain'][key]
        result += f'<tr><th scope="row">{arm}</th>' + ''.join(numeric_cell(name,['chain',key,metric],record[metric],language) for metric,_ in columns) + numeric_cell(name,['chain',key,'n'],record['n'],language) + '</tr>'
    result += '</tbody></table></div>' + paired_table(name,'paired',['primary/all: C − B','primary/all: C − A'], [('accord',('Mean agreement','Accord moyen')),('functional',('Final ≥ 0.99','Final ≥ 0,99'))],language)
    result += '<p>' + text(('C improves mean agreement, while final recovery at 0.99 falls relative to B. This is why a higher headline average cannot be presented as a solved capability.', 'C améliore l’accord moyen, tandis que la récupération finale à 0,99 recule par rapport à B. Une meilleure moyenne ne peut donc pas être présentée comme une capacité résolue.'),language) + '</p>' + provenance(name,language) + '</section>'
    result += '<section class="section"><h2>' + text(('Where the chain loses the required information', 'Où la chaîne perd l’information nécessaire'),language) + '</h2>' + bottleneck(language)
    result += '<div class="table-scroll"><table><caption>' + text(('C, primary regimes: measured recovery versus atom count · scale 0–1','C, régimes principaux : récupération mesurée selon le nombre d’atomes · échelle 0–1'),language) + '</caption><thead><tr><th scope="col">' + text(('Atoms','Atomes'),language) + '</th><th scope="col">' + text(('Individual recall','Rappel individuel'),language) + '</th><th scope="col">' + text(('All present in pool','Tous présents dans le pool'),language) + '</th><th scope="col">' + text(('All retained','Tous conservés'),language) + '</th><th scope="col">n</th></tr></thead><tbody>'
    for depth in range(1,7):
        key=f'primary/C/n{depth}'; record=data['set_scaling'][key]
        result += f'<tr><th scope="row">{depth}</th>' + ''.join(numeric_cell(name,['set_scaling',key,metric],record[metric],language).replace('</td>',bar(record[metric])+'</td>') for metric in ('atom_recall','all_present','kept_all')) + numeric_cell(name,['set_scaling',key,'n'],record['n'],language) + '</tr>'
    result += '</tbody></table></div><p class="note">' + text(('“Present” uses functional agreement ≥ 0.95 for an individual primitive; it does not mean exact parameters. Complete-pool recovery and retained-set recovery differ. Marginal columns are not a causal waterfall, and these results are not perception benchmarks on real scenes.', '« Présent » utilise un accord fonctionnel ≥ 0,95 pour une primitive ; il ne signifie pas des paramètres exacts. Ensemble complet dans le pool et ensemble conservé diffèrent. Ces colonnes marginales ne sont pas une cascade causale, et ces résultats ne sont pas des benchmarks de perception de scènes réelles.'),language) + '</p></section>'
    result += '<section class="section"><h2>' + text(('Secondary dense regime: the bottleneck moves, it does not disappear','Régime dense secondaire : le blocage se déplace, il ne disparaît pas'),language) + '</h2><div class="table-scroll"><table><caption>' + text(('C, dense regime; secondary evidence, not the primary verdict','C, régime dense ; preuve secondaire, pas le verdict principal'),language) + '</caption><thead><tr><th scope="col">' + text(('Complete pool','Pool complet'),language) + '</th><th scope="col">' + text(('Candidate ≥ 0.97','Candidat ≥ 0,97'),language) + '</th><th scope="col">' + text(('Final ≥ 0.99','Final ≥ 0,99'),language) + '</th><th scope="col">n</th></tr></thead><tbody><tr>'
    result += ''.join(numeric_cell(name,['chain','dense/all/C',metric],data['chain']['dense/all/C'][metric],language) for metric in ('pool_all_functional','candidate_097','functional','n')) + '</tr></tbody></table></div><p>' + text(('More observations improve the pool and candidate structure, but joint parameter precision and final selection still lose successes. Controlled one-parameter perturbations bound tolerances; simultaneous errors accumulate. No learned component was tested in this milestone.', 'Davantage d’observations améliorent le pool et les candidats, mais précision conjointe et sélection finale perdent encore des réussites. Les perturbations contrôlées d’un seul paramètre bornent les tolérances ; les erreurs simultanées s’accumulent. Aucun composant appris n’a été testé dans ce jalon.'),language) + '</p></section>'
    result += '<section class="section"><h2>' + text(('Hypotheses and the preregistered decision','Hypothèses et décision préenregistrée'),language) + '</h2><p>' + text(('The report judges the full conjunction of criteria, not just a positive difference. HP1 fails the required deep-regime recall level; HP3 fails complete-set recovery. HP5 fails the high-accuracy final criterion. The decision therefore remains blocking.', 'Le rapport juge la conjonction complète des critères, pas seulement une différence positive. HP1 échoue au niveau de rappel requis en régime profond ; HP3 à la récupération complète ; HP5 au critère final de haute exactitude. La décision reste donc bloquante.'),language) + '</p>'
    report=(ROOT/'docs/rapport_high_precision_primitives.md').read_text()
    criteria = report.split('### 0.6 Hypotheses',1)[1].split('### 0.7',1)[0]
    recall_threshold = float(re.search(r'recall of C at depths 4–6 ≥ ([\d.]+)',criteria)[1])
    set_threshold = float(re.search(r'AND ≥ ([\d.]+) at depths 4–6',criteria)[1])
    result += '<div class="table-scroll"><table><caption>' + text(('C at depths 4–6: observed level versus required support threshold', 'C aux profondeurs 4–6 : niveau observé et seuil requis pour soutenir l’hypothèse'),language) + '</caption><thead><tr><th scope="col">ID</th><th scope="col">'+text(('Observed','Observé'),language)+'</th><th scope="col">'+text(('Required ≥','Requis ≥'),language)+'</th></tr></thead><tbody>'
    for hypothesis, metric, threshold in [('HP1','pool_recall_functional',recall_threshold),('HP3','pool_all_functional',set_threshold)]:
        result += '<tr><th scope="row">'+hypothesis+'</th>'+numeric_cell(name,['chain','primary/d4-6/C',metric],data['chain']['primary/d4-6/C'][metric],language)+'<td>'+number(threshold,language)+'</td></tr>'
    result += '</tbody></table></div><p class="note">'+text(('Required levels read from report section 0.6; the full support rule also requires positive paired intervals. Meeting one condition would not suffice.', 'Niveaux requis lus dans la section 0.6 du rapport ; la règle complète exige aussi des intervalles appariés positifs. Atteindre une seule condition ne suffirait pas.'),language)+'</p>'
    # Use the report's final hypothesis table verbatim, with explicit source language.
    table=report.split('## 7. Preregistered hypotheses',1)[1].split('## 8.',1)[0]
    result += '<details class="science-hypotheses"><summary>' + text(('All HP1–HP13 verdicts · canonical English report','Tous les verdicts HP1–HP13 · rapport canonique anglais'),language) + '</summary><div lang="en" class="table-scroll"><table><caption>Report section 7 — preregistered hypotheses</caption><thead><tr><th scope="col">ID</th><th scope="col">Result</th><th scope="col">Status</th></tr></thead><tbody>'
    for line in table.splitlines():
        if line.startswith('| HP'):
            cells=[cell.strip().replace('**','').replace('`','') for cell in line.strip('|').split('|')]
            result += '<tr><th scope="row">'+escape(cells[0])+'</th>'+''.join('<td>'+escape(cell)+'</td>' for cell in cells[1:])+'</tr>'
    result += '</tbody></table></div></details><p>' + source_link('docs/rapport_high_precision_primitives.md') + '</p><p class="note">' + text(('No 16-seed confirmatory campaign proposed; live sealed set untouched; no component enabled by default. The new_onebirth arm is POST-HOC / EXPLORATORY. Residual morphology is not informative; out-of-class detection and the validation power/safety trade-off remain unresolved.', 'Aucune campagne confirmatoire à 16 graines proposée ; jeu scellé actif intact ; aucun composant activé par défaut. Le bras new_onebirth est POST-HOC / EXPLORATORY. La morphologie résiduelle n’est pas informative ; détection hors classe et compromis puissance/sécurité de validation restent non résolus.'),language) + '</p></section>'
    return result


def flow(items, label):
    return f'<ol class="science-flow" aria-label="{escape(label,quote=True)}">' + ''.join('<li><strong>'+escape(title)+'</strong><p>'+escape(copy)+'</p></li>' for title,copy in items) + '</ol>'


def bottleneck(language):
    en=[('Perception / input','Synthetic binary observations; general real-scene perception is future research.'),('Primitive discovery','Recall falls under occlusion and composition.'),('Parameter estimation','Individual functional agreement does not guarantee joint precision.'),('Complete set','Every required atom must be present and retained.'),('Composition','Strong controlled result with exact atoms, up to six.'),('Validation → reconstruction','A candidate can still fail final selection or validation.')]
    fr=[('Perception / entrée','Observations binaires synthétiques ; perception générale de scènes réelles : recherche future.'),('Découverte des primitives','Le rappel baisse sous occlusion et composition.'),('Estimation des paramètres','Un accord individuel ne garantit pas la précision conjointe.'),('Ensemble complet','Chaque atome requis doit être présent et conservé.'),('Composition','Résultat contrôlé solide avec atomes exacts, jusqu’à six.'),('Validation → reconstruction','Un candidat peut encore échouer à la sélection ou validation finale.')]
    return flow(fr if language=='fr' else en,text(('Conceptual bottleneck chain; not a quantified causal attribution','Chaîne conceptuelle des blocages ; pas une attribution causale chiffrée'),language))


def architecture(language):
    result=head('/architecture/research-map',language,('A hierarchy of mechanisms and research tracks. Conceptual arrows are reading relationships, not proof that all modules form one operational model.', 'Une hiérarchie des mécanismes et axes de recherche. Les flèches conceptuelles guident la lecture ; elles ne prouvent pas l’intégration de tous les modules en un modèle opérationnel.'))
    result+='<section class="section"><h2>'+text(('The conceptual loop','La boucle conceptuelle'),language)+'</h2>'
    en=[('Environment / world','Entities, properties and relations define the state.'),('Observation / measurement','A context and operator produce a measurement outcome.'),('World state / belief','Compatible states and evidence express uncertainty.'),('Transformations','Propose explicit changes with preconditions and invariants.'),('Composition / reasoning / planning','Build structured operations and evaluate possible consequences.'),('Evaluation / verification','Compare against observations or the relevant validity criterion.'),('Accept / reject / rollback','Commit supported changes; preserve the reference on rejection.'),('Learning / consolidation','Reuse or adapt validated structures within measured limits.')]
    fr=[('Environnement / monde','Entités, propriétés et relations définissent l’état.'),('Observation / mesure','Un contexte et un opérateur produisent un résultat de mesure.'),('État du monde / croyance','États compatibles et preuves représentent l’incertitude.'),('Transformations','Proposer des changements explicites avec préconditions et invariants.'),('Composition / raisonnement / planification','Construire des opérations structurées et évaluer leurs conséquences.'),('Évaluation / vérification','Comparer aux observations ou au critère de validité pertinent.'),('Accepter / rejeter / retour arrière','Consolider les changements soutenus ; préserver la référence en cas de rejet.'),('Apprentissage / consolidation','Réutiliser ou adapter les structures validées dans les limites mesurées.')]
    result+=flow(fr if language=='fr' else en,text(('Conceptual architecture, not a unified deployed pipeline','Architecture conceptuelle, pas une chaîne déployée unifiée'),language))+'</section>'
    result+='<section class="section"><h2>WorldState ≠ Observation ≠ BeliefState</h2><p>'+text(('The world is the latent object. A measurement is a context-dependent recorded outcome. A belief describes states still compatible with the evidence; it is not the world itself.', 'Le monde est l’objet latent. Une mesure est un résultat enregistré dépendant du contexte. Une croyance décrit les états encore compatibles avec les preuves ; elle n’est pas le monde lui-même.'),language)+'</p>'
    result+=flow([(key,text(copy,language)) for key,copy in [('WorldState',('Latent state','État latent')),('MeasurementOperator',('Context, observable, units and noise','Contexte, observable, unités et bruit')),('MeasurementOutcome',('Recorded outcome, not full truth','Résultat enregistré, pas vérité complète')),('CompatibleStateSet',('Several worlds can explain the same outcome','Plusieurs mondes peuvent expliquer le même résultat')),('Belief update',('Update uncertainty using the evidence','Mettre à jour l’incertitude à partir des preuves'))]],'World → measurement → compatible states → belief')+'<p>'+source_link('core/measurement.py')+' · '+source_link('tests/test_measurement.py')+'</p></section>'
    result+='<section class="section"><h2>'+text(('Primitives → transformations → compositions','Primitives → transformations → compositions'),language)+'</h2>'+flow([('A, B',text(('Small geometric predicates: disc, half-plane, band…','Petits prédicats géométriques : disque, demi-plan, bande…'),language)),('T(A)',text(('Transform coordinates or parameters with declared invariants.','Transformer coordonnées ou paramètres avec des invariants déclarés.'),language)),('A ∧ B / A ∨ B / A ⊕ B',text(('Boolean composition forms a structured function.','La composition booléenne forme une fonction structurée.'),language)),('Equivalence class',text(('Group candidates by their behaviour on observations. Same observed function need not mean the same syntax.','Regrouper les candidats par comportement observé. Même fonction observée ne signifie pas même syntaxe.'),language))],'Primitive composition')+'<p>'+source_link('core/boundary_program.py')+' · '+source_link('core/deep_compose.py')+'</p></section>'
    result+=transaction(language)
    groups=[
        (('Implemented symbolic path','Parcours symbolique implémenté'), ('Commands → executor → world/entities/relations → predefined physics → answer. Algebra represents changes to equations as explicit objects: subtracting the same term from both sides preserves equality. Conditions return TRUE, FALSE or UNKNOWN; unknown is not false. Planning searches state transitions toward a goal and keeps speculative branches separate. These are implemented small-domain mechanisms, not a general reasoning benchmark.','Commandes → exécuteur → monde/entités/relations → physique prédéfinie → réponse. L’algèbre représente les changements d’équations par des objets explicites : soustraire le même terme des deux côtés préserve l’égalité. Les conditions retournent TRUE, FALSE ou UNKNOWN ; inconnu ne signifie pas faux. La planification recherche des transitions vers un objectif et sépare les branches spéculatives. Ce sont des mécanismes implémentés à domaine restreint, pas un benchmark de raisonnement général.'), ('language/command.py','core/executor.py','core/world.py','core/algebra.py','core/reasoning.py','core/planner.py','tests/test_algebra.py','tests/test_reasoning_core.py','tests/test_planner.py')),
        (('Adaptive structure / learning experiments','Structure adaptative / expériences d’apprentissage'), ('Model state, structural learning, dynamic neural components, plasticity, lifelong adaptation and certificates are separate research mechanisms.','État du modèle, apprentissage structurel, composants neuronaux dynamiques, plasticité, adaptation continue et certificats sont des mécanismes de recherche distincts.'), ('core/model_state.py','core/learning.py','core/plasticity.py','core/safe_plasticity.py','core/lifelong.py','core/continuum.py','core/generative_certificate.py','tests/test_learning.py','tests/test_lifelong.py')),
        (('Language / evidence / temporal belief','Langage / preuves / croyance temporelle'), ('Command grammar differs from language experiments. Source reliability weights evidence; epistemic provenance distinguishes what supports an answer. TemporalBeliefState separates event time from ingestion time, combines reliability with persistence, and computes filtered and smoothed beliefs. A late report may revise a past belief; it does not retroactively change the world. These are bounded prototypes, not an integrated general reasoner.','La grammaire de commandes diffère des expériences de langage. La fiabilité pondère les preuves ; la provenance épistémique distingue ce qui soutient une réponse. TemporalBeliefState sépare temps de l’événement et temps de réception, combine fiabilité et persistance, et calcule croyances filtrées et lissées. Un rapport tardif peut réviser une croyance passée ; il ne modifie pas rétroactivement le monde. Ces prototypes sont bornés, pas un raisonneur général intégré.'), ('core/language_core.py','core/grounded.py','core/source_reliability.py','core/entitlement.py','core/temporal_belief.py','tests/test_temporal.py','tests/test_epistemic.py')),
        (('Discovery / reconstruction programme','Programme de découverte / reconstruction'), ('Structural reconstruction leads to deep candidate generation, primitive discovery, validated optimization, fragment attribution, equivalence-class composition, latent-set recovery and high-precision estimation. All current end-to-end claims remain experimental or blocked.','La reconstruction structurelle mène à génération profonde, découverte de primitives, optimisation validée, attribution des fragments, composition par classes, récupération latente et haute précision. Les capacités actuelles de bout en bout restent expérimentales ou bloquées.'), ('core/active_structure.py','core/deep_generation.py','core/deep_primitives.py','core/deep_fit.py','core/deep_fragments.py','core/deep_ownership.py','core/deep_compose.py','core/deep_e2e.py','core/deep_lpr.py','core/deep_hp.py','tests/test_deep_compose.py','tests/test_deep_hp.py')),
    ]
    result+='<section class="section"><h2>'+text(('Explore the code by research family','Explorer le code par famille de recherche'),language)+'</h2>'
    for title,copy,sources in groups:
        result+='<details class="science-family"><summary>'+text(title,language)+'</summary><p>'+text(copy,language)+'</p><ul>'+''.join('<li>'+source_link(path)+'</li>' for path in sources)+'</ul></details>'
    return result+'</section><section class="section"><h2>'+text(('The current discovery chain','La chaîne de découverte actuelle'),language)+'</h2>'+bottleneck(language)+'<a class="text-link" href="/research/programme">'+text(('Follow the experiments →','Suivre les expériences →'),language)+'</a></section>'


def transaction(language):
    labels=text((('State A','Proposal','Evaluation','Accepted → State B','Rejected → rollback → State A'), ('État A','Proposition','Évaluation','Acceptée → état B','Rejetée → retour arrière → état A')),language)
    result='<section class="section transaction-study" data-transaction><h2>'+text(('A proposal can be rejected without losing the reference','Une proposition peut être rejetée sans perdre la référence'),language)+'</h2><p>'+text(('Illustration of a transaction, not a live benchmark. The proposed move is held separately; validation decides whether to adopt it. A guardrail protects a reference but cannot make a wrong criterion correct.','Illustration d’une transaction, pas benchmark en direct. Le déplacement proposé reste séparé ; la validation décide de son adoption. Un garde-fou protège une référence mais ne corrige pas un mauvais critère.'),language)+'</p>'
    result+=flow([(labels[0],'x = 0'),(labels[1],'x = 1'),(labels[2],text(('Independent criterion','Critère indépendant'),language)),(labels[3],'x = 1'),(labels[4],'x = 0')],text(('Both outcomes remain visible without JavaScript','Les deux issues restent visibles sans JavaScript'),language))
    result+='<div class="transaction-controls" hidden><button type="button" data-decision="accept">'+text(('Accept proposal','Accepter la proposition'),language)+'</button><button type="button" data-decision="reject">'+text(('Reject and roll back','Rejeter et revenir'),language)+'</button></div><div class="transaction-scene" hidden><span class="transaction-reference">A · x = 0</span><span class="transaction-point" aria-hidden="true">●</span><span>B · x = 1</span></div><p role="status" data-transaction-result data-accept="'+escape(labels[3],quote=True)+'" data-reject="'+escape(labels[4],quote=True)+'"></p><p>'+source_link('docs/architecture_validated_transformation.md')+' · '+source_link('core/deep_fit.py')+'</p></section>'
    return result


def reproducibility(language):
    snapshot=json.loads((ROOT/'app/test_inventory.json').read_text())
    result='<section class="section"><h2>'+text(('Test scope and reproducibility','Portée des tests et reproductibilité'),language)+'</h2><p><strong>'+number(snapshot['declared_methods'],language)+'</strong> '+text(('statically declared test methods in tracked test_*.py files. This is a generated source inventory, not an executed or passing test total; dynamically generated cases are not counted.','méthodes de test déclarées statiquement dans les fichiers suivis test_*.py. Cet inventaire est généré depuis les sources ; ce n’est pas un total de tests exécutés ou réussis, et les cas générés dynamiquement ne sont pas comptés.'),language)+'</p><p>'+source_link('tests')+' · '+source_link('app/test_inventory.json')+' · '+source_link('Makefile')+'</p><details class="artifact-hash"><summary>'+text(('Test inventory fingerprint','Empreinte de l’inventaire des tests'),language)+'</summary><code>'+snapshot['source_sha256']+'</code></details>'
    result+='<h3>'+text(('Development ≠ confirmatory ≠ sealed','Développement ≠ confirmatoire ≠ scellé'),language)+'</h3><p>'+text(('Development keys support tuning and new comparisons with declared disjoint keys. Confirmatory campaigns require a frozen method and preregistered multi-seed criteria. The live deep v2 sealed set uses a private salt and has never been evaluated; v1 is retired, while earlier milestones have consumed historical sealed sets. Do not inspect or tune on live sealed tasks.','Les clés de développement servent au réglage et aux nouvelles comparaisons avec clés disjointes déclarées. Une campagne confirmatoire exige une méthode figée et des critères multi-graines préenregistrés. Le jeu deep v2 actif utilise un sel privé et n’a jamais été évalué ; v1 est retiré, et les anciens jalons ont consommé leurs jeux historiques. Ne pas inspecter ni régler sur les tâches scellées actives.'),language)+'</p><p>'+source_link('docs/sealed_sets.md')+'</p><pre><code>make help\nmake frontend-test\nmake check\nmake test  # research dependencies required\nmake deep-hp-test\nmake deep-hp-report</code></pre><p>'+text(('Report code version, dependencies, platform, precision, seeds, partition, compute budget and artifact fingerprints. Unit tests do not reproduce an experimental campaign. The README records known platform-dependent numerical divergences; they are not evidence of a universal passing suite.','Documenter version du code, dépendances, plateforme, précision, graines, partition, budget et empreintes. Les tests unitaires ne reproduisent pas une campagne. Le README signale des divergences numériques selon la plateforme ; cela ne démontre pas une suite universellement réussie.'),language)+'</p></section>'
    return result


def open_resources(language):
    paths=['README.md','CONTRIBUTING.md','LICENSE','SECURITY.md','CITATION.cff','docs/architecture_wce.md','docs/hypotheses_deep_compositional_discovery.md']
    return '<section class="section"><h2>'+text(('Code, contribution and citation','Code, contribution et citation'),language)+'</h2><p>'+text(('The current scientific code is public under Apache-2.0. Read the contribution rules, report security issues through SECURITY, and cite the project using CITATION.cff. Scientific reports are canonical English sources.','Le code scientifique courant est public sous Apache-2.0. Lire les règles de contribution, signaler les problèmes de sécurité via SECURITY et citer le projet avec CITATION.cff. Les rapports scientifiques canoniques sont en anglais.'),language)+'</p><div class="resource-links">'+''.join(source_link(path) for path in paths)+'</div><a class="text-link" href="/research/programme">'+text(('Browse the scientific reports →','Parcourir les rapports scientifiques →'),language)+'</a></section>'


def render_science(route,language):
    return {'/research/programme':programme,'/research/deep-composition':composition,'/research/high-precision':high_precision,'/architecture/research-map':architecture}[route](language)

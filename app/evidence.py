"""Read-only, server-rendered views of the three already-public research artifacts."""
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / 'results'
METRICS = {
 'retention': ('Rétention', 'Retention'),
 'average_accuracy': ('Exactitude moyenne', 'Average accuracy'),
 'new_task': ('Nouvelle tâche', 'New task'),
 'forgetting': ('Oubli', 'Forgetting'),
 'backward_transfer': ('Transfert arrière', 'Backward transfer'),
 'calibration_ece': ('Erreur de calibration (ECE)', 'Calibration error (ECE)'),
 'total_memory_bytes': ('Mémoire déclarée (octets)', 'Reported memory (bytes)'),
 'reuse_ratio': ('Ratio de réutilisation', 'Reuse ratio'),
 'neural_params': ('Paramètres neuronaux', 'Neural parameters'),
}


METRIC_GUIDES = {
 'retention': ('Exactitude finale moyenne sur les tâches retenues pour l’évaluation. Dans ce protocole, retention et average_accuracy ont la même définition ; ce ne sont pas deux preuves indépendantes.', 'Final mean accuracy on tasks retained for evaluation. In this protocol, retention and average_accuracy have the same definition; they are not two independent pieces of evidence.'),
 'average_accuracy': ('Moyenne des exactitudes finales des tâches non remplacées. Elle ne décrit pas une exactitude générale sur des tâches inconnues.', 'Mean final accuracy on tasks that have not been superseded. It does not describe general accuracy on unknown tasks.'),
 'new_task': ('Exactitude moyenne mesurée juste après l’apprentissage de chaque tâche. Elle distingue l’acquisition initiale de la performance finale.', 'Mean accuracy measured immediately after learning each task. It distinguishes initial acquisition from final performance.'),
 'forgetting': ('Baisse moyenne positive entre acquisition et évaluation finale, sur les tâches retenues. Une valeur plus faible indique moins d’oubli dans ce flux.', 'Mean positive drop from acquisition to final evaluation on retained tasks. A lower value indicates less forgetting on this stream.'),
 'backward_transfer': ('Différence moyenne entre performance finale et performance à l’acquisition. Une valeur négative indique une baisse ; une valeur positive indique une amélioration sur les tâches retenues.', 'Mean difference between final and acquisition performance. A negative value indicates a drop; a positive value indicates improvement on retained tasks.'),
 'calibration_ece': ('Écart pondéré entre confiance et exactitude, par groupes de confiance définis dans le code. Une erreur plus faible indique un meilleur accord dans cette évaluation, pas une garantie de vérité.', 'Weighted gap between confidence and accuracy, using confidence bins defined in the code. A lower error indicates closer agreement in this evaluation, not a guarantee of truth.'),
 'total_memory_bytes': ('Somme des postes mémoire en octets déclarés par la méthode. Ce compteur ne mesure pas toute la mémoire du processus ni les coûts matériels complets.', 'Sum of memory entries in bytes reported by the method. This counter measures neither total process memory nor complete hardware costs.'),
 'reuse_ratio': ('Part des étapes d’apprentissage classées reuse, compose ou reactivate. Un ratio élevé ne suffit pas à établir une meilleure performance.', 'Share of learning steps classified as reuse, compose or reactivate. A high ratio alone does not establish better performance.'),
 'neural_params': ('Nombre de paramètres neuronaux déclaré par la méthode. Il ne décrit pas à lui seul la mémoire, le temps d’entraînement ou la qualité du modèle.', 'Number of neural parameters reported by the method. It does not by itself describe memory, training time or model quality.'),
}


HYPOTHESIS_GUIDES = {
 'H1': ('Entrelacer recherche et expérience', 'Alterner recherche de programmes et observations améliore-t-il la récupération des structures, à budget comparable ?', 'Interleave search and experiments', 'Does alternating program search and observations improve structure recovery at a comparable budget?'),
 'H2': ('Choisir des requêtes informatives', 'La politique de gain d’information réduit-elle mieux l’incertitude structurelle que des requêtes aléatoires ?', 'Choose informative queries', 'Does the information-gain policy reduce structural uncertainty more effectively than random queries?'),
 'H3': ('Regrouper par comportement', 'Comparer les fonctions permet-il de retrouver plus de familles que comparer leur syntaxe, sans trop de faux regroupements ?', 'Group by behaviour', 'Does comparing functions recover more families than comparing syntax, without excessive false groupings?'),
 'H4': ('Vérifier malgré le bruit', 'Les observations de contrôle détectent-elles les règles fausses sans rejeter trop de règles correctes ?', 'Verify despite noise', 'Do control observations detect false rules without rejecting too many correct rules?'),
 'H5': ('Accepter les règles correctes', 'Le test de stabilité distingue-t-il les programmes corrects des cas non représentables, aux niveaux de bruit testés ?', 'Accept correct rules', 'Does the stability test distinguish correct programs from unrepresentable cases at the tested noise levels?'),
 'H6': ('Guider les recherches profondes', 'Le guidage réduit-il le nombre d’évaluations de candidats tout en conservant un taux de récupération comparable ?', 'Guide deeper searches', 'Does guidance reduce candidate evaluations while maintaining a comparable recovery rate?'),
 'H7': ('Réutiliser après transformation', 'Une représentation canonique améliore-t-elle la réutilisation après rotation, translation, changement d’échelle ou réflexion ?', 'Reuse after transformation', 'Does a canonical representation improve reuse after rotation, translation, scaling or reflection?'),
 'H8': ('Généraliser hors de la région testée', 'Les expériences structurelles améliorent-elles l’extrapolation par rapport aux requêtes aléatoires ?', 'Generalise beyond the tested region', 'Do structural experiments improve extrapolation compared with random queries?'),
 'H9': ('Limiter les corrections locales', 'Limiter l’application d’une correction réduit-il la dégradation hors région, tout en préservant la performance dans la région ?', 'Bound local corrections', 'Does restricting a correction’s application reduce degradation outside the region while preserving performance inside it?'),
 'H10': ('Retenir sous budget strict', 'Le système complet conserve-t-il davantage de compétences que les experts seuls, sans dépasser le budget de capacité ?', 'Retain under a strict budget', 'Does the full system retain more skills than experts alone without exceeding the capacity budget?'),
}


def artifact(name):
    raw=(RESULTS/name).read_bytes()
    return json.loads(raw),sha256(raw).hexdigest()


def number(value,language):
    text=f'{value:,.4f}'.rstrip('0').rstrip('.')
    return text.replace(',','\u202f').replace('.',',') if language=='fr' else text


def provenance(name,digest,language):
    label='Artefact source' if language=='fr' else 'Source artifact'
    hash_label='Empreinte complète du fichier' if language=='fr' else 'Full file fingerprint'
    return f'<div class="artifact-provenance"><a href="/research/{name}">{label} ↗</a><code>{escape(name)}</code><details class="artifact-hash"><summary>SHA-256 / {digest[:12]}…</summary><p>{hash_label}</p><code>{digest}</code></details></div>'


def heading(title,intro):
    return f'<div class="evidence-heading"><p class="eyebrow">WCE / EVIDENCE RECORD</p><h2>{title}</h2><p>{intro}</p></div>'


def predictive(language):
    fr=language=='fr';data,digest=artifact('predictive_coverage.json')
    title='La couverture dépend de la source.' if fr else 'Coverage depends on the source.'
    intro=('Valeurs de l’artefact, sur une échelle de 0 à 1. Couverture prédictive et couverture crédible sont deux mesures distinctes ; aucune des deux n’est une exactitude générale du modèle.' if fr else 'Artifact values on a 0–1 scale. Predictive and credible coverage are distinct measurements; neither is general model accuracy.')
    labels=('Prédictive','Crédible') if fr else ('Predictive','Credible')
    rows=[]
    for source in ('all','database','sensor','document','speaker'):
        record=data[source];p=record['predictive'];c=record['credible']
        title_source={'all':('Ensemble','All'),'database':('Base de données','Database'),'sensor':('Capteur','Sensor'),'document':('Document','Document'),'speaker':('Locuteur','Speaker')}[source][not fr]
        chart=f'<svg viewBox="0 0 420 48" preserveAspectRatio="none" aria-hidden="true"><path d="M0 12H420M0 36H420" class="coverage-track"/><path d="M0 12H{p*420:.3f}" class="coverage-predictive"/><path d="M0 36H{c*420:.3f}" class="coverage-credible"/></svg>'
        rows.append(f'<div class="coverage-row" data-source="{source}" data-predictive="{p}" data-credible="{c}" data-n="{record["n"]}"><div><strong>{title_source}</strong><span>n = {record["n"]}</span></div>{chart}<div class="coverage-values"><span>{labels[0]} <strong>{number(p,language)}</strong></span><span>{labels[1]} <strong>{number(c,language)}</strong></span></div></div>')
    guide=('La couverture indique la fréquence à laquelle un intervalle contient la proportion de réponses correctes dans ce protocole. L’intervalle crédible porte sur le paramètre de fiabilité ; l’intervalle prédictif inclut aussi la variabilité des réponses futures. Le champ n est l’effectif agrégé du résumé, pas le nombre total d’exemples d’entraînement.' if fr else 'Coverage measures how often an interval contains the proportion of correct answers in this protocol. The credible interval concerns the reliability parameter; the predictive interval also includes variability in future answers. The n field is the summary’s aggregated count, not the total number of training examples.')
    return '<section class="section evidence-record">'+heading(title,intro)+f'<div class="metric-guide"><p>{guide}</p></div><div class="coverage-legend"><span>{labels[0]}</span><span>{labels[1]}</span></div><div class="coverage-axis"><span>0</span><span>{"0,5" if fr else "0.5"}</span><span>1</span></div>'+''.join(rows)+provenance('predictive_coverage.json',digest,language)+'</section>'


def metric_panel(methods,metric,language,active):
    fr=language=='fr';entries=[(name,record[metric]) for name,record in methods.items()]
    low=min(0,min(record['ci95'][0] for _,record in entries))
    upper=max(1,max(record['ci95'][1] for _,record in entries))
    def x(value):return 10+400*(value-low)/(upper-low)
    plots=[];rows=[]
    for method,record in entries:
        mean=record['mean'];lo,hi=record['ci95'];n=record['n']
        chart=f'<svg viewBox="0 0 420 38" preserveAspectRatio="none" aria-hidden="true"><path class="estimate-track" d="M10 19H410"/><path class="estimate-interval" d="M{x(lo):.3f} 19H{x(hi):.3f}M{x(lo):.3f} 13V25M{x(hi):.3f} 13V25"/><path class="estimate-point" d="M{x(mean):.3f} 14V24"/></svg>'
        plots.append(f'<div class="estimate-row" data-method="{escape(method)}" data-metric="{metric}" data-estimate="{mean}" data-ci-low="{lo}" data-ci-high="{hi}" data-n="{n}"><code>{escape(method)}</code>{chart}<strong>{number(mean,language)}</strong></div>')
        rows.append(f'<tr><th scope="row"><code>{escape(method)}</code></th><td>{number(mean,language)}</td><td>[{number(lo,language)} ; {number(hi,language)}]</td><td>{n}</td></tr>')
    title=METRICS[metric][not fr]
    guide=METRIC_GUIDES[metric][not fr]
    unit=('octets' if fr else 'bytes') if metric=='total_memory_bytes' else ('paramètres' if fr else 'parameters') if metric=='neural_params' else ('valeur de la métrique' if fr else 'metric value')
    visible='' if active else ' hidden'
    return f'<div class="metric-panel" data-metric-panel="{metric}"{visible}><div class="metric-guide"><p class="eyebrow">{('LIRE CET INDICATEUR' if fr else 'READING THIS METRIC')}</p><p>{escape(guide)}</p></div><div class="estimate-axis"><span>{number(low,language)}</span><span>{unit}</span><span>{number(upper,language)}</span></div>'+''.join(plots)+'<details class="evidence-table"><summary>'+('Valeurs, intervalles et effectifs' if fr else 'Values, intervals and sample sizes')+'</summary><div class="table-scroll"><table><caption>'+escape(title)+'</caption><thead><tr><th scope="col">'+('Méthode' if fr else 'Method')+'</th><th scope="col">'+('Moyenne' if fr else 'Mean')+'</th><th scope="col">'+('IC à 95 %' if fr else '95% CI')+'</th><th scope="col">n</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></details></div>'


def continual(language):
    fr=language=='fr';data,digest=artifact('continuum_summary.json');methods=data['methods']
    options=''.join(f'<option value="{key}">{names[not fr]}</option>' for key,names in METRICS.items())
    panels=''.join(metric_panel(methods,key,language,key=='retention') for key in METRICS)
    title='Comparer plusieurs dimensions.' if fr else 'Compare multiple dimensions.'
    intro=('Neuf méthodes, neuf métriques. Les repères indiquent les moyennes et les segments les intervalles à 95 % fournis par l’artefact. L’ordre suit le fichier, sans classement global ; les coûts et les protocoles doivent être examinés ensemble.' if fr else 'Nine methods, nine metrics. Markers show means and segments the artifact’s reported 95% intervals. Order follows the file, without an overall ranking; costs and protocols must be examined together.')
    return '<section class="section evidence-record">'+heading(title,intro)+'<div class="evidence-controls"><label for="evidence-metric">'+('Métrique' if fr else 'Metric')+f'</label><select id="evidence-metric">{options}</select><span class="evidence-current" aria-live="polite">{METRICS["retention"][not fr]}</span></div>'+panels+'<p class="note">'+('La mémoire est la métrique déclarée par ce protocole, pas la mémoire vive du serveur web. Les mesures ne sont pas un benchmark contre les LLM. Sans JavaScript, la rétention reste lisible ; toutes les métriques sont dans le JSON.' if fr else 'Memory is the metric reported by this protocol, not web-server RAM. These measurements are not a benchmark against LLMs. Without JavaScript, retention remains readable; all metrics are in the JSON.')+'</p>'+provenance('continuum_summary.json',digest,language)+'</section>'


def discovery(language):
    fr=language=='fr';data,digest=artifact('active_summary.json');hypotheses=data['hypotheses']
    statuses={'SOUTENUE':('Soutenue','Supported','supported'),'RÉFUTÉE':('Réfutée','Refuted','refuted'),'NON CONCLUANTE':('Non concluante','Inconclusive','inconclusive')}
    counts=Counter(record['verdict'] for record in hypotheses.values())
    stats=''.join(f'<div><strong>{counts[status]}</strong><span>{labels[not fr]}</span></div>' for status,labels in statuses.items())
    cards=[]
    for key,record in hypotheses.items():
        label=statuses[record['verdict']]
        guide=HYPOTHESIS_GUIDES.get(key, (key, '', key, ''))
        question_title, question=guide[:2] if fr else guide[2:]
        raw=escape(json.dumps(record,ensure_ascii=False,indent=2))
        cards.append(f'<details class="hypothesis-card verdict-{label[2]}" data-hypothesis="{escape(key)}" data-verdict="{escape(record["verdict"])}"><summary><strong><span class="hypothesis-id">{escape(key)}</span><span class="hypothesis-title">{escape(question_title)}</span></strong><span>{label[not fr]}</span><span aria-hidden="true">+</span></summary><div><p class="hypothesis-question">{escape(question)}</p><p>'+('Champs exacts de cette hypothèse dans l’artefact :' if fr else 'Exact artifact fields for this hypothesis:')+f'</p><pre><code>{raw}</code></pre></div></details>')
    title='Tous les verdicts restent visibles.' if fr else 'Every verdict remains visible.'
    intro=('Dix identifiants propres au protocole active_summary.json. Un verdict soutenu est un résultat dans ce dispositif expérimental ; il n’établit pas une supériorité générale. Dépliez une hypothèse pour inspecter ses champs bruts.' if fr else 'Ten identifiers belonging to the active_summary.json protocol. A supported verdict is a result within this experimental setup, not general superiority. Expand a hypothesis to inspect its raw fields.')
    filters=[('all','Tous les verdicts','All verdicts'),('supported','Soutenus','Supported'),('refuted','Réfutés','Refuted'),('inconclusive','Non concluants','Inconclusive')]
    controls='<div class="verdict-filters" hidden role="group" aria-label="'+('Filtrer les hypothèses' if fr else 'Filter hypotheses')+'">'+''.join(f'<button class="filter" type="button" data-verdict-filter="{key}" aria-pressed="{str(key=="all").lower()}" aria-controls="hypothesis-results">{a if fr else b}</button>' for key,a,b in filters)+'<p role="status" class="verdict-filter-count" data-label="'+('hypothèses affichées' if fr else 'hypotheses shown')+'"></p></div>'
    legend='<div class="verdict-guide"><h3>'+('Interpréter un verdict' if fr else 'Interpreting a verdict')+'</h3><p>'+('Soutenu : le critère du protocole est atteint dans cette expérience. Réfuté : le critère d’échec est atteint. Non concluant : les conditions de succès ou d’échec ne permettent pas de trancher. Ces étiquettes sont celles du fichier source ; cette page ne recalcule pas les verdicts.' if fr else 'Supported: the protocol criterion is met in this experiment. Refuted: the failure criterion is met. Inconclusive: the success and failure conditions do not resolve the question. These labels come from the source file; this page does not recalculate verdicts.')+'</p><a class="text-link" href="/research/hypotheses_active_structure.md">'+('Lire le protocole préenregistré ↗' if fr else 'Read the preregistered protocol (French) ↗')+'</a></div>'
    return '<section class="section evidence-record">'+heading(title,intro)+legend+'<div class="verdict-counts">'+stats+'</div>'+controls+'<div class="hypothesis-grid" id="hypothesis-results">'+''.join(cards)+'</div>'+provenance('active_summary.json',digest,language)+'</section>'


def render_evidence(route,language):
    renderer={'/research/measurement':predictive,'/research/continual':continual,'/research/discovery':discovery}.get(route)
    return renderer(language) if renderer else ''
